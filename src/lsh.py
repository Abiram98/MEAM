import math
import sys

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

                avg += matrix_sim(self.bucket[i].matrix(9, self.bucket[j].matrix())

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

class LSHStore:
    def __init__(self):
        self.idx
        self.__fingerprint_sim_threshold = 0.9
        self.__buckets = list()
        self.__total_size = 0

    def add(self, element):
        found = False
        most_similar = 0
        most_similar_idx = -1

        for i in range(len(self.__buckets)):
            sim = matrix_sim(element.matrix(), self.__buckets[i].fingerprint())

            if (sim > most_similar):
                most_similar = sim
                most_similar_idx = i

                if (sim > self.__fingerprint_sim_threshold):
                    found = True

        if (found):
            self.__buckets[most_similar_idx].add(element.matrix())

        else:
            b = Bucket()
            b.add(element.matrix())
            self.__buckets.append(b)

        avg_fingerprints_sims = __avg_fingerprints_similarities()
        self.__fingerprint_sim_threshold = max(self.__fingerprint_sim_threshold, avg_fingerprints_sims)

    def __avg_fingerprints_similarities(self):
        avg = 0

        for i in range(len(self.__buckets)):
            for j in range(len(self.__buckets)):
                if (i == j):
                    continue

                avg += matrix_sim(self.__buckets[i].fingerprint(), self.__buckets[j].fingerprint())

        return avg / (pow(len(self.__buckets), 2) - len(self.__buckets))

    def search(self, matrix):
        highest_sim = 0
        highest_sim_idx = -1

        for i in range(len(self.__buckets)):
            if (matrix_sim(matrix, self.__buckets[i].fingerprint()) > highest_sim):
                highest_sim = matrix_sim(matrix, self.__buckets[i].fingerprint())
                highest_sim_idx = i

        best_element_sim = 0
        best_element = None

        for i in range(len(self.__buckets[highest_sim_idx].elements())):
            if (matrix_sim(matrix, self.__buckets[highest_sim_idx].elements()[i]) > best_element_sim):
                best_element_sim = matrix_sim(matrix, self.__buckets[highest_sim_idx].elements()[i])
                best_element = self.__buckets[highest_sim_idx].elements()[i]

        return best_element

if __name__ == "__main__":
    m1 = [[1, 2, 3], [3, 2, 1], [1, 1, 1]]
    m2 = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
    m3 = [[3, 3, 3], [2, 2, 2], [1, 1, 1]]
    print("Sim: " + str(matrix_sim(m1, m2)))

    b = Bucket()
    print("Uninitialized bucket fingerprint: " + str(b.fingerprint()))

    b.add(m1)
    b.add(m2)
    print("Fingerprint with two elements: " + str(b.fingerprint()))

    b.add(m2)
    print("Fingerprint with three elements: " + str(b.fingerprint()))
