import random as rnd
from tools.images.filters.edgeFilter import EdgeFilter
from tools.images.filters.hsvFilter import HsvFilter
from tools.images.filters.GenomeClass import Genome
from sklearn.model_selection import KFold
import numpy as np
from tensorflow import keras
import gc


class FilterGeneticAlgorithm:
    
    def __init__(self, X_train=None, Y_train=None, X_test=None, Y_test=None, model=None, loss=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.model=model
        self.loss=loss
    
    
    def generate_Selection(self):   
        return [rnd.randint(range[0],range[1]) for range in init_ranges]
    
    
    def generate_Population(self, size:int):
        population=[Genome([0,179,0,255,0,255,0,0,0,0,5,1,1,100,200,0],self.fitness([0,179,0,255,0,255,0,0,0,0,5,1,1,100,200,0]))]
        for _ in range(size-1):
            filter=self.generate_Selection() 
            population.append(Genome(filter,self.fitness(filter)))
        
        return population 
    
    
    
    def fitness(self, filter:list): 
        X_train=self.get_newData(self.X_train,filter)
        evaluations=0
    
        kf = KFold(n_splits=10, random_state=42, shuffle=True)
        print("--------------------------------------------------------------------------------")

        for train_index, test_index in kf.split(X_train):
            
            model= keras.models.clone_model(self.model)
            
            model.compile(optimizer="adam", loss=self.loss, metrics=['accuracy'])
            
            train_index, test_index = train_index.tolist(), test_index.tolist()
            x_train, x_test, y_train, y_test = np.array(X_train)[train_index], np.array(X_train)[test_index], np.array(self.Y_train)[train_index], np.array(self.Y_train)[test_index]
            
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            
            evaluations+=model.evaluate(x_test,y_test)[1]
            
            gc.collect()
        
        print(f"filter {filter} fit: {evaluations/10}")
        return evaluations/10
    
    
    
    def selection_pair(self, population:list[Genome]):
        return rnd.choices(
            population=population,
            weights=[genome.fit for genome in population],
            k=2
        )

    
    
    def single_point_crossover(self, a: Genome, b: Genome):
        length = len(a.filter)
            
        if length < 2:
            return a, b
            
        p = rnd.randint(1, length-1)
        population=[a,b]
            
        app=rnd.choices(
                population=population,
                weights=[genome.fit for genome in population],
                k=2
            )
            
        c1=Genome(a.filter[0:p]+b.filter[p:15]+[app[0].filter[15]],self.fitness(a.filter[0:p]+b.filter[p:15]+[app[0].filter[15]]))
            
        c2= Genome(b.filter[0:p]+a.filter[p:15]+[app[1].filter[15]],self.fitness(b.filter[0:p]+a.filter[p:15]+[app[1].filter[15]]))
            
        return  c1,c2
    
    
    
    def mutation(self, genome: Genome, probability: float=0.05):
        new_filter = []
            
        for ind,i in enumerate(genome.filter):
            if float(rnd.random()) < probability:
                min,max=ranges[genome.filter.index(i)]
                if ind not in [1,3,5]:
                    new_filter.append(abs(i-rnd.randint(min,max)))
                else:
                    new_filter.append(rnd.randint(new_filter[ind-1],ranges[ind][1]))
                    
            else:
                new_filter.append(i)
                
        return Genome(new_filter,self.fitness(new_filter))
    
    
    
    def apply_Filter(self, image, filter:list):
        hsv_filter=HsvFilter(hMin=filter[0], hMax=filter[1], sMin=filter[2], sMax=filter[3], vMin=filter[4], vMax=filter[5], sAdd=filter[6], sSub=filter[7], vAdd=filter[8], vSub=filter[9])
        canny_filter=EdgeFilter(kernelSize=filter[10], erodeIter=filter[11], dilateIter=filter[12], canny1=filter[13], canny2=filter[14])
        
        filtered_image=hsv_filter.apply_hsv_filter(image,hsv_filter)
        
        if filter[15]==1:
            filtered_image=canny_filter.apply_edge_filter(image,canny_filter)
        
        if filter[15]==2:
            filtered_image=canny_filter.apply_edge_filter(filtered_image,canny_filter)
        
        return filtered_image
    
    
    
    def get_newData(self, data, filter:list):
        return [ self.apply_Filter(image,filter) for image in data]
    
    
    
    def run(self, generation_limit: int, population_size: int):

        population = self.generate_Population(population_size)

        for i in range(generation_limit):

            population = sorted(
                population,
                key=lambda genome: genome.fit,
                reverse=True
            )

            print('#################################################################################################################################################################') 
            print(f"\ngeneration: {i} best filter :{population[0].filter} fit: {population[0].fit}\n")
            print('#################################################################################################################################################################')

            next_generation = population[0:2]

            for j in range(int(len(population)/2)-1):
                parent = self.selection_pair(population)
                a, b = self.single_point_crossover(parent[0], parent[1])
                next_generation += [self.mutation(a), self.mutation(b)]

            population = next_generation

            
        population = sorted(
            population,
            key=lambda genome: genome.fit,
            reverse=True
        )

        return population[0]

# set filter rules(Define range of parameters)
ranges = (
    (0, 179),
    (0, 179),
    (0, 255),
    (0, 255),
    (0, 255),
    (0, 255),
    (0, 255),
    (0, 255),
    (0, 255),
    (0, 255),
    (0, 30),
    (0, 5),
    (0, 5),
    (0, 200),
    (0, 500),
    (0, 2)
)

init_ranges = (
    (0, 60),
    (80, 179),
    (0, 85),
    (105, 255),
    (0, 85),
    (105, 255),
    (0, 200),
    (0, 200),
    (0, 200),
    (0, 200),
    (0, 30),
    (0, 5),
    (0, 5),
    (0, 200),
    (0, 500),
    (0, 2)
)
