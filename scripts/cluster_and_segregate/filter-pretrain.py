import os, re
import pandas as pd
from argparse import ArgumentParser

def build_cluster_dataframe(clstr_path):
    id_re = re.compile(r">(\S+?)\.{3}")
    len_re = re.compile(r"(\d+)aa,")
    rows, cluster_id = [], None

    with open(clstr_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">Cluster"):
                cluster_id = int(line.split()[-1])
                continue
            m_id, m_len = id_re.search(line), len_re.search(line)
            if m_id and m_len:
                rows.append({
                    "id": m_id.group(1), 
                    "cluster": cluster_id, 
                    "length": int(m_len.group(1))
                })

    return pd.DataFrame(rows)


if __name__ == "__main__" :
    
    parser = ArgumentParser(description="Filter pretrain dataset")
    parser.add_argument("--cluster", help="Path to pretrain .clstr file", required=True)
    parser.add_argument("--pretrain", help="Path to pretrain CSV file", required=True)
    parser.add_argument("--finetune", help="Path to clustered finetune CSV file", required=True)
    parser.add_argument("--out", help="Path to output CSV file", required=True)

    args = parser.parse_args()

    # load pretrain, finetune and parse pretrain cluster data
    pretrain_df = pd.read_csv(args.pretrain)
    finetune_df = pd.read_csv(args.finetune)
    pretrain_cluster_df = build_cluster_dataframe(args.cluster)

    final_pretrain_df = pd.merge(pretrain_df[['id', 'sequence']], pretrain_cluster_df, on="id")

    # Get finetune val & test sequences
    finetune_test_sequences = finetune_df.loc[finetune_df['split'] == 'test', 'sequence'].unique().tolist()
    finetune_val_sequences = finetune_df.loc[finetune_df['split'] == 'val', 'sequence'].unique().tolist()
    finetune_seq_to_exclude = finetune_test_sequences + finetune_val_sequences

    # find which clusters to remove in the pretrain data
    clusters_to_remove = final_pretrain_df.loc[final_pretrain_df['sequence'].isin(finetune_seq_to_exclude), 'cluster'].values.tolist()

    # remove clusters and save
    final_pretrain_df[~final_pretrain_df['cluster'].isin(clusters_to_remove)].drop(columns=['length', 'cluster']).to_csv(args.out, index=False)
