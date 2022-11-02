import random
#from numpy.lib.function_base import append
import tsplib95
import networkx as nx
import math
import sys
from threading import Thread

import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class Gui(QMainWindow):
    def __init__(self):
        """ Main window of GUI.
        """
        super().__init__()
        self.init_graph = nx.Graph()
        self.result_graph = nx.Graph()
        self.max_y = 1000
        self.running = False
        self.f_path = ""

        self.setWindowTitle('TSP Genetic Algorithm')
        self.setMinimumSize(QSize(800, 500))
        self.content = QWidget()
        layout = QGridLayout()

        self.input_generations = QLineEdit('1000')
        self.input_elites_p = QLineEdit('0.2')
        self.input_pop_size = QLineEdit('100')
        self.input_mutation_rate = QLineEdit('0.15')
        self.input_random_cities = QLineEdit('30')
        self.button_file = QPushButton('Select file')
        self.button_file.clicked.connect(self.open_file_name_dialog)
        self.button_random = QPushButton('Start random')
        self.button_random.clicked.connect(self.threading_rand)
        self.button_tspf = QPushButton('Start from file')
        self.button_tspf.clicked.connect(self.threading_tspf)
        self.label_running = QLabel("")
        self.label_file_path = QLabel("")
        self.label_result = QLabel("")

        layout.addWidget(QLabel('Number of generations:'), 1, 0)
        layout.addWidget(QLabel('Size of population:'), 2, 0)
        layout.addWidget(QLabel('Mutation rate:'), 3, 0)
        layout.addWidget(QLabel('Percantage of individuals, that survive into next generation:'), 4, 0)
        layout.addWidget(QLabel('Number of randomly generated cities:'), 5, 0)
        layout.addWidget(QLabel('Select .tsp file:'), 6, 0)
        layout.addWidget(self.input_generations, 1, 1)
        layout.addWidget(self.input_pop_size, 2, 1)
        layout.addWidget(self.input_mutation_rate, 3, 1)
        layout.addWidget(self.input_elites_p, 4, 1)
        layout.addWidget(self.input_random_cities,5,1)
        layout.addWidget(self.button_file, 6, 1)
        layout.addWidget(self.label_file_path,7,0,1,2)
        layout.addWidget(self.button_random, 8, 0)
        layout.addWidget(self.button_tspf, 8, 1)
        layout.addWidget(self.label_running, 9,0,1,2)
        layout.addWidget(self.label_result, 10, 0, 1, 2)

        self.figure1 = Figure()
        self.canvas1 = FigureCanvasQTAgg(self.figure1)
        self.axes1 = self.figure1.add_subplot(211)
        self.axes1.set_title('Lowest length in generations')
        self.axes2 = self.figure1.add_subplot(212)
        nx.draw(self.result_graph,ax=self.axes2)
        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas1, 0, 2, 10, 30)

        self.content.setLayout(layout)
        self.setCentralWidget(self.content)


    def open_file_name_dialog(self):
        """When the button is clicked, the open dialog appears to allow the user to choose .tsp file.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","tsp Files (*.tsp)", options=options)
        if fileName:
            self.f_path = fileName
            self.label_file_path.setText("File selected: "+self.f_path)

    def threading_rand(self):
        """Creates another thread for solving TSP of randomly generated cities.
        """
        if not self.running:
            self.cities = self.generatePoints(int(self.input_random_cities.text()))
            t1=Thread(target=self.start_tsp)
            t1.setDaemon(True)
            t1.start()

    def threading_tspf(self):
        """Creates another thread for solving TSP from .tsp file.
        """
        try:
            problem = tsplib95.load(self.f_path)
            self.cities = problem.node_coords
            if not self.running:
                t1=Thread(target=self.start_tsp)
                t1.setDaemon(True)
                t1.start()
        except:
            pass

    def start_tsp(self):
        """Starts the solution of TSP with Genetic Algorithm.
        """
        self.result_graph.clear()
        self.result_graph.clear_edges()
        self.running = True
        self.label_running.setText("Running...")
        self.tsp = GenAlg(self)
        self.best = self.tsp.best_result
        self.generations_n = int(self.input_generations.text())
        self.tsp.find_optimum(len(self.cities),self.cities,int(self.input_generations.text()),float(self.input_mutation_rate.text()),int(self.input_pop_size.text()),float(self.input_elites_p.text()))      
        if self.tsp.graph_generation[-1] != self.generations_n:
            self.tsp.graph_generation.append(self.generations_n)
            self.tsp.graph_cost.append(self.tsp.graph_cost[-1])
        self.draw_result_graph()
        self.running = False
        self.label_running.setText("Completed!")

    def draw_result_graph(self):
        """Draws the graph of best length with respect to generation and the graph of best route.
        """
        route_edges = [(self.best.cities_index[i]+1,self.best.cities_index[i+1]+1) for i in range(len(self.best.cities_index)-1)]
        self.result_graph.add_nodes_from(self.cities.keys())
        self.result_graph.add_edges_from(route_edges)
        self.axes2.clear()
        nx.draw(self.result_graph,pos=self.cities,edgelist=route_edges,edge_color='r',ax=self.axes2,node_size=30)
        
        self.axes1.clear()
        self.axes1.set_title('Lowest length in generations')
        self.axes1.set_xlim([0, self.generations_n])
        self.axes1.set_ylim([0, self.max_y])
        self.axes1.plot(self.tsp.graph_generation, self.tsp.graph_cost)
        self.canvas1.draw()

        self.label_result.setText("Lowest length: " + str(self.tsp.best_result.len))


    def generatePoints(self, n):
        """Generates number cities with random coordinates.
        """
        cities = {}
        for i in range(n):
            cities[i+1] = (random.uniform(0,1000), random.uniform(0,1000))
        return cities

    
class GenAlg:
    def __init__(self,gui = None):
        self.population = []
        self.total_len = 0
        self.tot_fit = 0
        self.fitness_all = 0
        self.parents = []
        self.generation = 0
        self.no_change = 0
        self.gui = gui
        self.best_result = None
        self.graph_generation = []
        self.graph_cost = []

    def generate_init_population(self,n):
        """Generates initial population.
        """
        for i in range(n):
            self.population.append(Chromosome(self.n_nodes))
            random.shuffle(self.population[i].cities_index)
        self.calculate_fitness()
        self.normalize_fitness()
    
    def calculate_fitness(self):
        """Calculates fitness function of all chromosomes in population.
        """
        self.total_len = 0
        self.fitness_all = 0
        for ch in self.population:
            ch.lifespan += 1
            if ch.lifespan >= 50:
                random.shuffle(ch.cities_index)
                ch.lifespan = 0
            ch.len_path_all(self.cities)
            self.total_len += ch.len
            if ch.len < self.best_result.len:
                self.best_result.len = ch.len
                self.best_result.cities_index = [city for city in ch.cities_index]
                self.best_result.cities_index.append(self.best_result.cities_index[0])
                print("Generation: %d: %.4f" % (self.generation, self.best_result.len))
                self.graph_generation.append(self.generation)
                self.graph_cost.append(self.best_result.len)
                if len(self.graph_cost)==1:
                    self.gui.max_y = self.graph_cost[0]
                self.gui.draw_result_graph()  
            self.fitness_all += 1/ch.len
        
    def normalize_fitness(self):
        """Normalizes fitness score of chromosomes to be in interval <0;1>.
        """
        for ch in self.population:
            ch.fitness = (1/ch.len) / self.fitness_all
            self.tot_fit += ch.fitness
    
    def roulette_wheel_selection(self):
        """Selects randomly parent.
        """
        rand_select = random.random()
        prob = 0
        for ch in self.population:
            prob += ch.fitness
            if rand_select <= prob:
                return ch
    
    def generate_new_population(self,elites):
        """Generates new generation of individuals. Percantage of elites survives into another generation.
        """
        self.generation += 1
        new_population = []
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        elites_n = math.floor(elites*len(self.population))
        for i in range(elites_n):
            new_population.append(self.population[i])
        for _ in range(int(len(self.population))-elites_n):
            p1 = self.roulette_wheel_selection()
            p2 = self.roulette_wheel_selection()
            child = self.order_crossover(p1,p2)
            self.mutate_swap_random(child)
            new_population.append(child)
        self.population = new_population
        self.calculate_fitness()
        self.normalize_fitness()

    def order_crossover(self,parent1,parent2):
        """Performs corssover on parent1 and parent2 to create new child.
        """
        first = random.randint(0,self.n_nodes-2)
        second = random.randint(first+1,self.n_nodes-1)

        child1 = Chromosome(self.n_nodes)
        child1.cities_index = [-1 for i in range(len(parent1.cities_index))]
        j = second
        for i in range(first,second):
            child1.cities_index[i] = parent1.cities_index[i]
        for i in range(second,len(parent1.cities_index)):
            while(True):
                if (parent2.cities_index[j] not in child1.cities_index):
                    child1.cities_index[i] = parent2.cities_index[j]
                    j += 1
                    if j >= len(parent2.cities_index):
                        j = 0
                    break
                j += 1
                if j >= len(parent2.cities_index):
                    j = 0
        for i in range(first):
            while(True):
                if (parent2.cities_index[j] not in child1.cities_index):
                    child1.cities_index[i] = parent2.cities_index[j]
                    break
                j += 1
        return child1

    def mutate_swap_random(self,chrom):
        """Performs muttion by swaping random genes.
        """
        if random.random() <= self.mutation_rate:
            first_change = random.randint(0,self.n_nodes-1)
            second_change = random.randint(0,self.n_nodes-1)
            chrom.cities_index[first_change], chrom.cities_index[second_change] = chrom.cities_index[second_change], chrom.cities_index[first_change]


    def mutate_swap_neighbors(self,chrom):
        """Performs mutation by swaping adjacent genes.
        """
        if random.random() <= self.mutation_rate:
            first_change = random.randint(0,self.n_nodes-1)
            second_change = first_change + 1
            if (second_change > len(chrom.cities_index)-1):
                second_change = first_change - 1
            chrom.cities_index[first_change], chrom.cities_index[second_change] = chrom.cities_index[second_change], chrom.cities_index[first_change]


    def find_optimum(self,nodes,city_list,generations,mutation_r,init_pop,p_elites):
        """For number of generations the algorithm tries to find the solution of TSP for given cities. 
        """
        self.n_nodes = nodes
        self.best_result = Chromosome(self.n_nodes)
        self.best_result.len = sys.maxsize
        self.gui.best = self.best_result
        self.cities = city_list
        self.mutation_rate = mutation_r
        self.generate_init_population(init_pop)
        for i in range(generations):
            self.no_change += 1
            self.generate_new_population(p_elites)
    
class Chromosome:
    def __init__(self,n):
        self.cities_index = list(range(0,n))
        self.len = 0
        self.fitness = 0
        self.lifespan = 0
    
    def len_path_two(self,start,end):
        """Returns lenght between two points.
        """
        return math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
    
    def len_path_all(self,cities):
        """Computes length of the whole route.
        """
        self.len = 0
        for i in range(len(cities)-1):
            start_i = self.cities_index[i]
            end_i = self.cities_index[i+1]
            self.len += self.len_path_two(cities[start_i+1],cities[end_i+1])
        self.len += self.len_path_two(cities[len(cities)],cities[1])

def main():
    app = QApplication(sys.argv)
    window = Gui()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()