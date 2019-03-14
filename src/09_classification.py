import libs.libs_rysy_python.rysy as rysy


dataset = rysy.DatasetBinary("datasets/mnist_9_9/training.bin", "datasets/mnist_9_9/testing.bin")

'''
classification_fc = rysy.ClassificationExperiment(dataset, "networks/mnist_0_fc/")
classification_fc.run()
'''


classification_cnn = rysy.ClassificationExperiment(dataset, "networks/mnist_1_cnn/")
classification_cnn.run()


'''
classification_cnn = rysy.ClassificationExperiment(dataset, "networks/mnist_2_dense_cnn/")
classification_cnn.run()
'''
