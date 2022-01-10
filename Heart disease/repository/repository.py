import csv


class Repository:

    def load_data(self, filename):
        '''
        Load data from CSV file
        :param filename: path to the file
        :return: the header of the CSV file and the rows as an array
        '''

        file = open(filename, 'r')
        csv_reader = csv.reader(file)

        header = next(csv_reader)

        rows = []
        for row in csv_reader:
            rows.append(row)

        file.close()

        return header, rows
