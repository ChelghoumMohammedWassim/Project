import random as rnd
from tools.images.crop.GenomeClass import Genome
from sklearn.model_selection import KFold
import numpy as np
from tensorflow import keras
import gc
from PIL import Image

class CropGeneticAlgorithm:
    
    def __init__(self, X_train=None, Y_train=None, X_test=None, Y_test=None, model=None, loss=None, IMAGE_SIZE=(224,224)):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.model=model
        self.loss=loss
        self.IMAGE_SIZE=IMAGE_SIZE
    
    
    def generate_Selection(self):
        width, height = self.IMAGE_SIZE
        left=rnd.randint(0, width-1)
        top=rnd.randint(0, height-1)
        right=left+rnd.randint(0, width-left)
        bottom=top+rnd.randint(0, height-top)
        return [(top,bottom),(left, right)]
    
    
    def generate_Population(self, size:int):
        width, height=self.IMAGE_SIZE
        population=[Genome([(0,height),(0,width)], self.fitness([(0,height),(0,width)]))]
        for _ in range(size-1):
            crop_dimension=self.generate_Selection() 
            population.append(Genome(crop_dimension,self.fitness(crop_dimension)))
        
        return population 
    
    
    
    def fitness(self, filter): 
        X_train=self.get_newData(self.X_train,filter)
        evaluations=0
    
        kf = KFold(n_splits=10, random_state=42, shuffle=True)
        print("--------------------------------------------------------------------------------")

        for train_index, test_index in kf.split(X_train):
            
            model= keras.models.clone_model(self.model)
            
            model.compile(optimizer="adam", loss=self.loss, metrics=['accuracy'])
            
            train_index, test_index = train_index.tolist(), test_index.tolist()
            x_train, x_test, y_train, y_test = np.array(X_train)[train_index], np.array(X_train)[test_index], np.array(self.Y_train)[train_index], np.array(self.Y_train)[test_index]
            
            model.fit(x_train, y_train, epochs=1)
            
            evaluations+=model.evaluate(x_test,y_test)[1]
            
            gc.collect()
        
        print(f"crop {filter} fit: {evaluations/10}")
        return evaluations/10
    
    
    
    def selection_pair(self, population:list[Genome]):
        return rnd.choices(
            population=population,
            weights=[genome.fit for genome in population],
            k=2
        )

    
    
    def single_point_crossover(self, a: Genome, b: Genome):
        c = rnd.choice([True, False])
        if not c:
            return a,b
                
        c1= Genome([a.crop_dimension[0],b.crop_dimension[1]], self.fitness([a.crop_dimension[0],b.crop_dimension[1]]))
        c2= Genome([b.crop_dimension[0],a.crop_dimension[1]], self.fitness([b.crop_dimension[0],a.crop_dimension[1]]))
            
        return  c1,c2
    
    
    
    def mutation(self, genome: Genome, probability: float=0.05):
        width, height = self.IMAGE_SIZE
        
        left=genome.crop_dimension[1][0]
        right=genome.crop_dimension[1][1]
        top=genome.crop_dimension[0][0]
        bottom=genome.crop_dimension[0][1]
        
        if rnd.random()< probability:
            left= rnd.randint(0, width-1)
            if left< right:
                if rnd.random()< probability:
                    right=rnd.randint(left+1, width)
            else:
                right=rnd.randint(left+1, width)
        
        
        if rnd.random()< probability:
            left= rnd.randint(0, height-1)
            if top< bottom:
                if rnd.random()< probability:
                    right=rnd.randint(top+1, height)
            else:
                right=rnd.randint(top+1, height)
        
        
        if [(top,bottom),(left, right)]==genome.crop_dimension:
            return genome
        return Genome([(top,bottom),(left, right)], self.fitness([(top,bottom),(left, right)]))
    
    
    def crop_Image(self, image, crop_dimension):
        left = crop_dimension[0][0]
        right = crop_dimension[0][1]
        top = crop_dimension[1][0]
        bottom = crop_dimension[1][1]

        pil_img = Image.fromarray(image)

        cropped_img = pil_img.crop((left, top, right, bottom))

        resized_img = cropped_img.resize((224, 224))

        resized_img_np = np.array(resized_img)

        return resized_img_np
    
    
    
    def get_newData(self, data, crop_dimension:list):
        return [ self.crop_Image(image,crop_dimension) for image in data]
    
    
    
    def run(self, generation_limit: int, population_size: int):

        population = self.generate_Population(population_size)

        for i in range(generation_limit):

            population = sorted(
                population,
                key=lambda genome: genome.fit,
                reverse=True
            )

            print('#################################################################################################################################################################') 
            print(f"\ngeneration: {i} best crop :{population[0].crop_dimension} fit: {population[0].fit}\n")
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