import os, json
import pandas as pd
from argparse import ArgumentParser

def read_marlys_into_df(marlys_json_file_path):
    marlys_data = json.load(open(marlys_json_file_path, 'r'))
    marlys_formatted_data = [{
        'id' : data['id'], 
        'sequence' : data['sequence'], 
        'activity' : ";".join(data['activity']), 
        'databases' : ";".join(data['databases']), 
        **data['properties'
    ]} for data in marlys_data]

    marlys_df = pd.DataFrame(marlys_formatted_data)
    return marlys_df

def filter_marlys_df(marlys_df) :

    final_marlys_df = marlys_df[
        # No disulfide bonds
        (marlys_df['disulfide'] == 0) &
        # Length cap between 8-50
        (marlys_df['length'] >= 8) & 
        (marlys_df['length'] <= 50)
        ]

    return final_marlys_df

if __name__ == "__main__" :

    parser = ArgumentParser(description="Filter Marlys AMP dataset")
    parser.add_argument("--marlys", help="Path to Marlys JSON file", required=True)
    parser.add_argument("--out", help="CSV path to save output", required=True)
    parser.add_argument("--fasta", help="FASTA path to save output", default=None)

    args = parser.parse_args()

    marlys_df = read_marlys_into_df(args.marlys)
    filtered_marlys_df = filter_marlys_df(marlys_df)
    filtered_marlys_df['id'] = ["PT_"+str(i) for i in range(1, len(filtered_marlys_df)+1)]
    filtered_marlys_df[['id', 'sequence']].drop_duplicates().to_csv(args.out, index=False)

    if args.fasta is not None :
        fasta_content = ''
        for idx, row in filtered_marlys_df.iterrows() :
            fasta_content += f'>{row["id"]}\n{row["sequence"]}\n'

        open(args.fasta, "w").write(fasta_content)
