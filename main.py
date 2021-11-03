from Model.Model import FC_Densenet
from time import time, gmtime, strftime

if __name__ == "__main__":
    start_time = time()
    my_model = FC_Densenet(input_shape=(224, 224, 3), layers_per_block=(4, 5, 7, 10, 12, 15))
    my_model = my_model.create_model()
    print(strftime('%H:%M:%S', gmtime(time() - start_time)))
