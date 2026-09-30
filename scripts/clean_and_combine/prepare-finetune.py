import pandas as pd
from argparse import ArgumentParser
import os

def is_canonical(seq):
    return set(seq).issubset(set('ACDEFGHIKLMNPQRSTVWY'))

def filter_dbaasp_df(dbaasp_activity_file_path):

    dbaasp_raw_df = pd.read_csv(dbaasp_activity_file_path)

    # monomer filter
    monomer_dbaasp_df = dbaasp_raw_df[dbaasp_raw_df['complexity'] == 'Monomer']

    # 20 proteinogenic amino acids
    subset_dbaasp_df = monomer_dbaasp_df[monomer_dbaasp_df['sequence'].apply(is_canonical) == True]

    final_subset_dbaasp_df = subset_dbaasp_df[

        # length cap between 8-50 AA
        (subset_dbaasp_df['sequence'].apply(len) >= 8) & 
        (subset_dbaasp_df['sequence'].apply(len) <= 50) &
        # No N- or C-terminus
        (subset_dbaasp_df['nTerminus'].isna()) & 
        (subset_dbaasp_df['cTerminus'].isna()) &
        # no intra-chain bonds
        (subset_dbaasp_df['intrachainBondCount'] == 0)
    ]

    return final_subset_dbaasp_df


def filter_grampa_df(grampa_file_path):

    grampa_raw_df = pd.read_csv(grampa_file_path)

    final_filtered_grampa_df = grampa_raw_df[

        # No modifications
        (grampa_raw_df['modifications'] == '[]') &
        (grampa_raw_df['has_unusual_modification'] == False) &
        (grampa_raw_df['is_modified'] == False) & 
        # No C-terminal amidation
        (grampa_raw_df['has_cterminal_amidation'] == False) & 
        # 20 proteinogenic amino acids
        (grampa_raw_df['sequence'].apply(is_canonical) == True) & 
        # length cap between 8-50 AA
        (grampa_raw_df['sequence'].apply(len) >= 8) &
        (grampa_raw_df['sequence'].apply(len) <= 50)
    ]

    return final_filtered_grampa_df

def filter_dramp_df(general_file_path, antibacterial_file_path):

    dramp_general_df = pd.read_csv(general_file_path, sep='\t')
    dramp_antibacterial_df = pd.read_csv(antibacterial_file_path, sep='\t')

    dramp_antibacterial_subset = dramp_general_df[dramp_general_df['DRAMP_ID'].isin(dramp_antibacterial_df['DRAMP_ID'])]

    filtered_dramp_df = dramp_antibacterial_subset[

        # No C-terminal or N-terminal modification
        (dramp_antibacterial_subset['C-terminal_Modification'] == 'Free') &
        (dramp_antibacterial_subset['N-terminal_Modification'] == 'Free') &
        # No non-linear structures
        (dramp_antibacterial_subset['Linear/Cyclic/Branched'] == 'Linear') &
        # No other modifications
        (dramp_antibacterial_subset['Other_Modifications'] == 'Free') &
        # length cap between 8-50 AA
        (dramp_antibacterial_subset['Sequence_Length'] >= 8) &
        (dramp_antibacterial_subset['Sequence_Length'] <= 50)
    ]

    return filtered_dramp_df

if __name__ == "__main__" :
    parser = ArgumentParser(description="Prepare finetuning dataset:")
    parser.add_argument("--dbaasp", help="File path to DBAASP activity CSV", required=True)
    parser.add_argument("--grampa", help="File path to GRAMPA activity CSV", required=True)
    parser.add_argument("--dramp", help="File path to DRAMP data DIRECTORY", required=True)
    parser.add_argument("--out", help="File path to combined sequence CSV", required=True)
    parser.add_argument("--fasta", help="File path to combined sequence FASTA", default=None)

    args = parser.parse_args()

    filtered_dbaasp_df = filter_dbaasp_df(args.dbaasp)
    filtered_grampa_df = filter_grampa_df(args.grampa)
    filtered_dramp_df = filter_dramp_df(os.path.join(args.dramp, "general_amps.txt"), os.path.join(args.dramp, "Antibacterial_amps.txt"))

    print(f"Sequence set: DBAASP - {filtered_dbaasp_df['sequence'].nunique()}, GRAMPA - {filtered_grampa_df['sequence'].nunique()}, DRAMP - {filtered_dramp_df['Sequence'].nunique()}")
    sequence_set = list(set(filtered_dramp_df['Sequence'].unique().tolist() + filtered_grampa_df['sequence'].unique().tolist() + filtered_dbaasp_df['sequence'].unique().tolist()))
    pretrain_id_set = list(range(1, len(sequence_set)+1))
    print(f"Combined set has: {sequence_set.__len__()} sequences")

    sequence_df = pd.DataFrame(sequence_set, columns=['sequence'])
    sequence_df['id'] = ["FT_"+str(i) for i in pretrain_id_set]
    sequence_df[['id', 'sequence']].to_csv(args.out, index=False)

    if args.fasta is not None :
        fasta_content = ''
        for idx, row in sequence_df.iterrows() :
            fasta_content += f'>{row["id"]}\n{row["sequence"]}\n'

        open(args.fasta, "w").write(fasta_content)