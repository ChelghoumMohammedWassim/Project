import pandas as pd
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.filters import Filter
from weka.core.dataset import create_instances_from_matrices
from tools.numerical.GenomeClass import Genome
import numpy as np
import random as rnd


class NumericalGeneticAlgorithm:
    def __init__(self, train:pd.DataFrame, test:pd.DataFrame):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.rnd_seed = 42
        self.train = train
        self.test = test
        jvm.start(system_cp=True, packages=True, max_heap_size="512m")

    def generate_Selection(self, length):
        return rnd.choices([0, 1], k=length)

    def generate_Population(self, size, length):
        population=[]
        for _ in range(size):
            selection=self.generate_Selection(length)
            operation=rnd.randint(1, 3)
            population.append(Genome(selection= selection, operation=operation,fit= self.fitness(selection, operation, self.rnd_seed)))
        return population
    
    
    def fitness(self, genome_selection, genome_operation, rnd_seed):                
        data=self.get_data_from_genome(genome_selection, genome_operation, self.X_train)

        if len(data)>0:
            dataset = create_instances_from_matrices(np.array(data), np.array(self.Y_train), name="generated from matrices")
            dataset.class_is_last()  
            
            string_to_nominal_filter = Filter(classname="weka.filters.unsupervised.attribute.StringToNominal", options=["-R", "last"])
            string_to_nominal_filter.inputformat(dataset)
            dataset = string_to_nominal_filter.filter(dataset)
            
            cls = Classifier(classname="weka.classifiers.trees.J48")

            evaluation = Evaluation(dataset)
            evaluation.crossvalidate_model(cls, dataset, 2, Random(rnd_seed))
            return evaluation.percent_correct
        return 0

    
    def selection_pair(self, population):
        return rnd.choices(
            population=population,
            weights=[genome.fit
                     for genome in population],
            k=2
        )
        
    
    def single_point_crossover(self, a: Genome, b: Genome):
        length = len(a.selection)
        if length < 2:
            return a, b
        p = rnd.randint(1, length-1)
        
        first_selection= a.selection[0:p]+b.selection[p:]
        first_operation= rnd.choice([a.operation, b.operation])
        first_child=Genome(
                        first_selection,
                        first_operation,
                        self.fitness(genome_selection= first_selection, genome_operation= first_operation, rnd_seed= self.rnd_seed))
        
        
        second_selection= b.selection[0:p]+a.selection[p:]
        second_operation= rnd.choice([a.operation, b.operation])
        second_child=Genome(
                        second_selection,
                        second_operation,
                        self.fitness(genome_selection= second_selection, genome_operation= second_operation, rnd_seed= self.rnd_seed))
        
        return  first_child, second_child
    
    
    def mutation(self, genome: Genome, probability=0.05):
        operation=rnd.randint(1,3)
        
        if 0 in genome.selection:
            selection = []
            for i in genome.selection:
                if float(rnd.random()) < probability:
                    selection.append(abs(i-1))
                else:
                    selection.append(i)
            
            return Genome(selection, operation, self.fitness(selection, operation, self.rnd_seed))
        return Genome(genome.selection, operation, self.fitness(genome.selection, operation, self.rnd_seed))

    
    def get_data_from_genome(self, genome_selection, genome_operation,data_set):

        data = self.get_selected_Columns(genome_selection, data_set)
        
        if len(data)>0:
            new_colomn=[]
            if genome_operation == 1:
                new_colomn=self.addition(data)
            elif genome_operation == 2:
                new_colomn=self.multiplication(data)
            elif genome_operation== 3:
                new_colomn=self.average(data)
            
            
            if (max(new_colomn)-min(new_colomn))>0:
                new_colomn=[ (x-min(new_colomn))/(max(new_colomn)-min(new_colomn)) for x in new_colomn] 
            data.append(new_colomn)
            
            data = [list(x) for x in zip(*data)]
        
        return data
    
    
    def get_selected_Columns(self, selection, data_set):
        data = []
        for i in range(len(selection)):
            if selection[i] == 1:
                data.append(data_set[i])
        return data
    
   
    def addition(self, data):
        return [sum(column) for column in zip(*data)]

  
    def multiplication(self, data):
        new_data = []
        for i in range(len(data[0])):
            r = 1
            for j in range(len(data)):
                r = r*data[j][i]
            new_data.append(r)
        return new_data

    
    def average(self, data):
        column_sums = self.addition(data)
        return [sum / len(data[0]) for sum in column_sums]
    
    
    def get_X_Y(self, data):
        return data[0:len(data)-1], data.pop(-1)
    
    
    # def get_output(self, genome: Genome, data_set, label):
    #     data=self.get_data_from_genome(genome.selection, genome.operation, data_set)

    #     dataset = create_instances_from_matrices(np.array(data), np.array(label), name="generated from matrices")
    #     dataset.class_is_last()  
            
    #     string_to_nominal_filter = Filter(classname="weka.filters.unsupervised.attribute.StringToNominal", options=["-R", "last"])
    #     string_to_nominal_filter.inputformat(dataset)
    #     dataset = string_to_nominal_filter.filter(dataset)
    #     cls = Classifier(classname="weka.classifiers.trees.J48")
    #     evaluation = Evaluation(dataset)
    #     evaluation.crossvalidate_model(cls, dataset, 2, Random(self.rnd_seed))
    #     print(evaluation.summary())
        
    #     return data_set
        
    def get_output(self, genome: Genome, data_set,label):
        data = self.get_selected_Columns(genome.selection, data_set) 
        if len(data)>0:
            new_colomn=[]
            if genome.operation == 1:
                new_colomn=self.addition(data)
            elif genome.operation == 2:
                new_colomn=self.multiplication(data)
            elif genome.operation == 3:
                new_colomn=self.average(data)
                
            if (max(new_colomn)-min(new_colomn))>0:
                new_colomn=[ (x-min(new_colomn))/(max(new_colomn)-min(new_colomn)) for x in new_colomn] 
                
            data.append(new_colomn)
            data.append(label)
            data = [list(x) for x in zip(*data)]

            header = ["att "+str(i+1) for i in range(len(data[0])-1)]
            header.append("class")
            data.insert(0, header)
            return data
        return data
        

    
    def run(self, generation_limit: int, population_size: int, rnd_seed: int = 42):
        
        self.rnd_seed= rnd_seed
        self.X_train, self.Y_train = self.get_X_Y(self.train.T.values.tolist())
        self.X_test, self.Y_test = self.get_X_Y(self.test.T.values.tolist())

        
        population = self.generate_Population(
            population_size, len(self.X_train))
        
        for i in range(generation_limit):

            population = sorted(
                population,
                key=lambda genome: genome.fit,
                reverse=True
            )
            
            # print(f"Generation: {i} best solution acc: {population[0].fit}")
            
            next_generation = population[0:2]
            
            for _ in range(int(len(population)/2)-1):
                parent = self.selection_pair(population)
                a, b = self.single_point_crossover(parent[0], parent[1])
                next_generation += [self.mutation(a), self.mutation(b)]
                
            population = next_generation
            
        population = sorted(
                population,
                key=lambda genome: genome.fit,
                reverse=True
            )
        
        print(f"Best solution acc: {population[0].fit}")
        
        return self.get_output(population[0], self.X_train, self.Y_train), self.get_output(population[0], self.X_test, self.Y_test)