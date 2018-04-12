import csv

def read_seed(seed_size, seed_number):
    filepath = "./Random_Seeds/" + str(seed_size)
    filepath += "/seed_"
    filepath += str(seed_number)
    seed_list = []
    with open (filepath) as csv_file:
        seed_reader = csv.reader(csv_file, delimiter = ',')
        for row in seed_reader:
            for item in row:
                seed_list.append(int(item))
    return seed_list
