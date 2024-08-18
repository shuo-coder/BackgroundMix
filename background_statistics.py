import csv
from collections import defaultdict

file_path = 'background_samemean.csv'
keywords = ["class", "building", "Tree", "Parking space", "Road", "Grassland", "Exhibition hall"]

data_by_category = defaultdict(list)

with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        category = row[0] 
        data_by_category[category].append(row)

output_file_path = 'background_statistics.csv'

with open(output_file_path, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    for category, data in data_by_category.items():
        word_frequency = defaultdict(int)
        for row in data:
            words = row[-1].split('，')  
            for word in words:
                if word.strip():  
                    word_frequency[word.strip()] += 1
        
        row_to_write = [category]
        for keyword in keywords[1:]:
            
            row_to_write.append(word_frequency[keyword])
        
        writer.writerow(row_to_write)

