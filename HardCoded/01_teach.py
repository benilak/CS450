import random
import numpy as np


# Part 1
class Movie:
    def __init__(self, title="", year=0, runtime=0):
        self.title = title
        self.year = year
        if runtime < 0:
            self.runtime = 0
        else:
            self.runtime = runtime

    def __repr__(self):
        return "{} ({}) - {} mins".format(self.title, self.year, self.runtime)

    def hours_minutes(self):
        return int(self.runtime/60), self.runtime % 60


Shrek = Movie("Shrek", 2001, 100)
print(Shrek)
print(str(Shrek.hours_minutes()))


# Part 2
def create_list():
    movies = [Movie("The Goonies", 1985, 114), Movie("Titanic", 1997, 194), Movie("Avatar", 2009, 162),
              Movie("I Am Legend", 2007, 101)]
    return movies


movies150 = [m for m in create_list() if m.runtime > 150]

for m in movies150:
    print(m)

movies = create_list()
stars = {}
for m in movies:
    star = random.uniform(0, 5)
    stars[m.title] = star

for m in movies:
    print("{} - {:.2f} stars".format(m.title, stars[m.title]))


# Part 3
def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100

        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array


def main():
    for m in create_list():
        print(m)
    array = get_movie_data()

    np.set_printoptions(suppress=True)
    print("Array: \n" + str(array))

    print("rows: " + str(array.shape[0]))
    print("columns: " + str(array.shape[1]))
    print("first 2 rows: \n" + str(array[0:2]))
    print("last 2 columns: \n" + str(array[:, -2:]))

    print("1d array - 2nd column: \n" + str(array[:, 1]))


if __name__ == "__main__":
    main()
