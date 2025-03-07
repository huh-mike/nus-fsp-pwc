import csv
from typing import List, Dict, Union


def save_scraped_data_to_csv(scraped_data: List[Dict[str, Union[str, None, List[str]]]],
                             filename: str = 'scraped_data.csv'):
    """
    Save scraped data to a CSV file.

    :param scraped_data: List of dictionaries containing scraped data
    :param filename: Name of the output CSV file (default: 'scraped_data.csv')
    """
    # If the list is empty, return early
    if not scraped_data:
        print("No data to save.")
        return

    # Get all unique keys from the dictionaries
    fieldnames = set()
    for item in scraped_data:
        fieldnames.update(item.keys())

    # Convert set to sorted list for consistent column order
    fieldnames = sorted(list(fieldnames))

    # Write to CSV
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header
            writer.writeheader()

            # Write data rows
            for row in scraped_data:
                # Ensure all keys exist, convert None to empty string
                processed_row = {key: str(row.get(key, '')) if row.get(key) is not None else ''
                                 for key in fieldnames}
                writer.writerow(processed_row)

        print(f"Data successfully saved to {filename}")

    except IOError as e:
        print(f"Error writing to CSV file: {e}")