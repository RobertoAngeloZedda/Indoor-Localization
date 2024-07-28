from PIL import Image
import numpy as np

# USED TO EXTRACT DATASET, NOT NEEDED ANYMORE
'''with open('./dataset/new_dataset.csv', 'w') as output_file:
    with open('./dataset/training_list.csv', 'r') as file_training:
        for line in file_training.readlines():
            line_list = line.split(',')
            output_file.write(f'{line_list[0]},{line_list[-1]}')

    with open('./dataset/validation_list.csv', 'r') as file_validation:
        for line in file_validation.readlines():
            line_list = line.split(',')
            output_file.write(f'{line_list[0]},{line_list[-1]}')'''

def load_dataset(file_path, img_dir_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            line_list = line.split(',')
            dataset.append((open_image(img_dir_path, line_list[0]), int(line_list[-1][:-1])))
    return dataset

def open_image(dir_path, img_file_name):
    pil_image = Image.open(dir_path + img_file_name)
    img = np.asarray(pil_image)
    return img

def create_set(dataset, classes, limit):
    
    np.random.shuffle(dataset)

    # keeps track of labels chosen to not surpass limit
    X = []
    labels = []
    remaining = []
    count = [0 for _ in range(len(classes))]
    for x, y in dataset:
        if y in classes and count[classes.index(y)] < limit:
            X.append(x)
            labels.append(y)
            count[classes.index(y)] += 1
        else:
            remaining.append((x, y))

    X = np.asarray(X) / 255

    Y = []
    for y in labels:
        tmp = [[0] for _ in range(len(classes))]
        tmp[classes.index(y)] = [1]
        Y.append(tmp)
    Y = np.asarray(Y)

    return X, Y, remaining

if __name__ == '__main__':
    
    dataset = load_dataset('./dataset/new_dataset.csv', './dataset/images/')
    print('Dataset Loaded.\n')

    #print(len(dataset))

    x_train, y_train, remaining  = create_set(dataset, [0, 1, 2, 3], 200)
    x_test, y_test, _ = create_set(remaining, [0, 1, 2, 3], 50)

    print(x_test.shape)
    print(y_test.shape)