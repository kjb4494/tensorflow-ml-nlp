
def stop_word():
    extractor = []
    with open('./data_in/stop_word_01.txt') as file:
        for line in file.readlines():
            extractor.append(line.strip())

    with open('./data_in/stop_word_02.txt') as file:
        for line in file.readlines():
            word_list = line.split(',')
            extractor = extractor + word_list
    result = set(extractor)
    for line in result:
        print(line)


if __name__ == '__main__':
    stop_word()
