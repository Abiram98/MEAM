import math
import sys
import numpy as np
import pickle

# Cosine similarity
def cosine(vec1, vec2):
    if (len(vec1) != len(vec2)):
        raise "Vectors of different dimensions"

    dot = length1 = length2 = 0

    for i in range(len(vec1)):
        dot += vec1[i] * vec2[i]
        length1 += pow(vec1[i], 2)
        length2 += pow(vec2[i], 2)

    length1 = math.sqrt(length1)
    length2 = math.sqrt(length2)

    return dot / (length1 * length2)

# Matrix similarity by cosine similarity averaging
def matrix_sim(m1, m2):
    if (len(m1) != len(m2)):
        raise "Un-matching number of rows"

    avg = 0

    for i in range(len(m1)):
        avg += cosine(m1[i], m2[i])

    return avg / len(m1)

class Element:
    def __init__(self, matrix, label):
        self.__m = matrix
        self.__l = label

    def label(self):
        return self.__l

    def matrix(self):
        return self.__m

class Bucket:
    def __init__(self):
        self.__fingerprint = None
        self.bucket = list()

    def add(self, element):
        if (self.fingerprint == None):
            self.__fingerprint = element.matrix()

        self.bucket.append(element)
        self.__find_fingerprint()

    def __find_fingerprint(self):
        highest_sim = 0
        highest = self.bucket[0]

        for i in range(len(self.bucket)):
            avg = 0

            for j in range(len(self.bucket)):
                if (i == j):
                    continue

                avg += matrix_sim(self.bucket[i].matrix(), self.bucket[j].matrix())

            if (len(self.bucket) > 1):
                avg = avg / (len(self.bucket) - 1)

            if (avg > highest_sim):
                highest_sim = avg
                highest = self.bucket[i]

        self.__fingerprint = highest.matrix()

    def fingerprint(self):
        return self.__fingerprint

    def elements(self):
        return self.bucket

    # This does not clear the bucket fingerprint
    def clear(self):
        self.bucket = list()

class LSHStore:
    def __init__(self):
        self.__fingerprint_sim_threshold = 0.9
        self.__buckets = list()
        self.__total_size = 0
        self.__mem_optimized = False

    def add(self, element):
        most_similar = 0
        most_similar_idx = -1

        for i in range(len(self.__buckets)):
            sim = matrix_sim(element.matrix(), self.__buckets[i].fingerprint())

            if (sim > most_similar):
                most_similar = sim
                most_similar_idx = i

        if (most_similar > self.__fingerprint_sim_threshold):
            self.__buckets[most_similar_idx].add(element)

        else:
            b = Bucket()
            b.add(element)
            self.__buckets.append(b)

        if (len(self.__buckets) > 1):
            max_fingerprints_sims = self.__max_fingerprints_similarities()
            self.__fingerprint_sim_threshold = max(self.__fingerprint_sim_threshold, max_fingerprints_sims)

    def __max_fingerprints_similarities(self):
        max_sim = 0

        for i in range(len(self.__buckets)):
            for j in range(len(self.__buckets)):
                if (i == j):
                    continue

                sim = matrix_sim(self.__buckets[i].fingerprint(), self.__buckets[j].fingerprint())

                if (sim > max_sim):
                    max_sim = sim

        return max_sim

    def search(self, matrix):
        highest_sim = 0
        highest_sim_idx = -1

        for i in range(len(self.__buckets)):
            if (matrix_sim(matrix, self.__buckets[i].fingerprint()) > highest_sim):
                highest_sim = matrix_sim(matrix, self.__buckets[i].fingerprint())
                highest_sim_idx = i

        if (self.__mem_optimized):
            return [Element(self.__buckets[highest_sim_idx].fingerprint(), "fingerprint"), highest_sim]

        best_element_sim = 0
        best_element = None

        if (highest_sim_idx == -1):
            return None

        for i in range(len(self.__buckets[highest_sim_idx].elements())):
            if (matrix_sim(matrix, self.__buckets[highest_sim_idx].elements()[i].matrix()) > best_element_sim):
                best_element_sim = matrix_sim(matrix, self.__buckets[highest_sim_idx].elements()[i].matrix())
                best_element = self.__buckets[highest_sim_idx].elements()[i]

        return [best_element, best_element_sim]

    def mem_optimize(self):
        for bucket in self.__buckets:
            bucket.clear()

        self.__mem_optimized = True

    def bucket_sizes(self):
        sizes = list()

        for bucket in self.__buckets:
            sizes.append(len(bucket.elements()))

        return sizes

# Usage: <command> <data file> <database file>
if __name__ == "__main__":
    if (len(sys.argv) < 4):
        print("Usage: <command> <file>")
        exit(-1)

    DB_FILE = sys.argv[3]
    SIM_THRES = 0.9

    if (sys.argv[1] == "load"):
        data = np.load(sys.argv[2])
        store = LSHStore()

        for i in range(len(data)):
            store.add(Element(data[i], i))
            print("Progress: {prog:.2f}%".format(prog = (i / len(data)) * 100), end = "\r")

        with open(DB_FILE, "wb") as file:
            pickle.dump(store, file)

    elif (sys.argv[1] == "search"):
        store = None
        queries = np.load(sys.argv[2])

        with open(DB_FILE, "rb") as file:
            store = pickle.load(file)

        with open("output.csv", "w") as file:
            for i in range(len(data)):
                result_sim = store.search(queries[i])[1]
                is_song = 0

                if (result > SIM_THRES):
                    is_song = 1

                file.write(str(i) + "," + str(is_sone))

    else:
        print("Usage: Commands include 'load' and 'search'")
        exit(-1)
