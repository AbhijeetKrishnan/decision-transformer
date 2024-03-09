import csv
import sys


def extract_data_from_log(log_file_path):
    data = []
    itr = 1
    with open(log_file_path, "r") as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("DEF"):
                program = line.strip()
                reward = float(lines[i + 1].strip())
                data.append((itr, program, reward))
                i += 1
            elif line.startswith("Iteration"):
                itr += 1
            i += 1
    return data


def write_to_csv(data, csv_file_path):
    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["iteration", "program", "reward"])
        for row in data:
            csv_writer.writerow(row)


def all_tasks():
    tasks = ('cleanHouse', 'fourCorners', 'harvester', 'randomMaze', 'stairClimber', 'topOff')
    for task in tasks:
        log_file_path = f"dt-{task}.log"  # Replace with your log file path
        csv_file_path = f"dt-{task}.csv"  # Output CSV file path

        try:
            data = extract_data_from_log(log_file_path)
            write_to_csv(data, csv_file_path)
            print(f"Data written to {csv_file_path}")
        except FileNotFoundError:
            print(f"Could't find file: {log_file_path}")


def main():
    log_file_path = sys.argv[1]
    csv_file_path = sys.argv[2]
    try:
        data = extract_data_from_log(log_file_path)
        write_to_csv(data, csv_file_path)
        print(f'Data written to {csv_file_path}')
    except FileNotFoundError:
        print(f"Couldn't find file: {log_file_path}")


if __name__ == "__main__":
    main()
