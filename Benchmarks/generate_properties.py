import os
import sys
import csv
from haoyu_gen import main as card_main
from cheng_gen import main as index_main
from decima_gen import main as decima_main
from aurora_gen import main as aurora_main
from pensieve_gen import main as pensieve_main
from bloom_filter_gen import main as bloom_filter

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_properties.py <random seed>, default is 2024")
        seed = 2024
    else:
        seed = sys.argv[1]

    if not os.path.exists('vnnlib'):
        os.makedirs('vnnlib')

    csv_data = []
    print("Generating Index specifications, this may take several minutes...")
    csv_data.extend(index_main(seed))
    print("Generating cardinality specifications, this may take around ten minutes...")
    csv_data.extend(card_main(seed))
    print(f"Successfully generate {len(csv_data)} files!")
    decima_main(seed)
    aurora_main(seed)
    pensieve_main(seed)
    bloom_filter(seed)
    print(f"Total timeout is {sum([int(i[-1]) for i in csv_data])}")
    with open('instances.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
