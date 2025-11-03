
# !pip install Bio --quiet

import numpy as np
import pandas as pd
import os, re
from glob import glob
from Bio import SeqIO
from typing import Literal
from itertools import product
import gc
import shutil
from collections import defaultdict, Counter
from Bio.Seq import Seq
from joblib import Parallel, delayed
import warnings, logging
from tqdm import tqdm

warnings.filterwarnings('ignore')


logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(levelname)s - %(message)s')



# read train and test
train = pd.read_csv('data/Train.csv')
train_subjects = pd.read_csv('data/Train_Subjects.csv')
test = pd.read_csv('data/Test.csv')

train.shape, train_subjects.shape, test.shape


# add folder path
folder_path = 'data/'
train_files = glob(folder_path+'/TrainFiles*/*.fastq', recursive=True)
test_files = glob(folder_path+'/TestFiles/*.fastq', recursive=True)


len(test_files), len(train_files)


# __Helper functions__

def get_batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


def trim_read(record, min_quality=20, min_length=50):
    """
    Trims low-quality bases from both ends of a read
    Discards reads shorter than min_length after trimming
    
    :param record: BioPython record class
    :param min_quality: Mimimum quality threshold adter trimming
    :param min_length: Minimum length of trimmed read to keep
    :returns Trimmed seqRecord or None if read is discarded
    """
    qual = record.letter_annotations['phred_quality']
    seq = str(record.seq)
    # Trim low-quality bases from 5' end
    start = 0
    while start < len(qual) and qual[start] < min_quality:
        start += 1
    
    # Trim low-quality bases from 3' end
    end = len(qual) - 1
    while end >= start and qual[end] < min_quality:
        end -= 1
    
    if end - start + 1 < min_length:
        return None  # Too short after quality trimming
    else:
        return seq[start:end+1]


# kmer vocab
def build_kmer_vocab(k:int):
    vocab = sorted(map(lambda x: ''.join(x), list(product('AGCT', repeat=k))))
    vocab_idx_map = dict(zip(vocab, range(1, len(vocab)+1)))
    return vocab_idx_map

def trim_N(trimmed_seq):
    # Trim Ns from both ends
    seq = trimmed_seq
    start = 0
    end = len(seq) - 1
    while start <= end and seq[start] == 'N':
        start += 1
    while end >= start and seq[end] == 'N':
        end -= 1
    return seq[start:end+1]

def trim_low_quality_and_Ns(record, **kwargs):
    """
    Trim low-quality bases and Ns from both ends of a read.
    Discard reads with internal Ns or shorter than min_length after trimming.
    
    :param record: SeqRecord object with 'phred_quality' annotation.
    :param min_quality: Minimum quality threshold for trimming.
    :param min_length: Minimum length of trimmed read to keep.
    :Returns Trimmed SeqRecord or None if read is discarded.
    """
    min_quality= kwargs.get('min_quality', 20)
    min_length= kwargs.get('min_length', 50)
    trim_n = kwargs.get('trim_n', False)
    trimmed_seq= trim_read(record, min_quality, min_length)
    if trimmed_seq is None:
        return None
    if trim_n: # if N should be trimmed
        trimmed_seq = trim_N(trimmed_seq)
    return trimmed_seq


def read_fastq(file_path, **kwargs):
    """Retrieves and trims read sequences from Fastq file"""
    for record in SeqIO.parse(file_path, "fastq"):
        seq = trim_low_quality_and_Ns(record)
        if seq is not None:
            yield seq


def generate_kmer_counts(seqs, k):
    """Generates Kmer counts"""
    vocab = build_kmer_vocab(k)
    if not isinstance(seqs, list):
        seqs = [seqs]
    
    kmer_counter = Counter(vocab.keys())
    for seq in seqs:
        if seq is None:
            continue
        kmer_counter.update(generate_kmer(seq, k))
    return kmer_counter


def generate_kmer(seq, k):
    """Yields kmers from a given sequence"""
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        if 'N' not in kmer:
            yield kmer


def preprocess_fastq2(filepath, k):
    """
    Preprocesses Fastq file based on list of kmers given
    Returns a Dataframe containing counts for each kmer
    """
    def convert_counts_to_dataframe(kmer_seqs, id):
        res = pd.DataFrame(kmer_seqs.values(), index=kmer_seqs.keys(), columns=[id]).astype(np.uint32)
        return res
    try:
        pattern = 'ID_[A-Za-z0-9]+(?!^.fastq)'
        ID = re.match(pattern, os.path.basename(filepath)).group()
        read_seqs = list(read_fastq(filepath))
        
        if not isinstance(k, (list, tuple)):
            k = [k]

        kmer_count_list = []
        for i in k:
            kmer_counts = generate_kmer_counts(read_seqs, i)
            # filter kmers greater than 1 and convert to dataframe
            # kmer_counts = dict(filter(lambda x: x[1] > 1, kmer_counts.items()))
            if len(kmer_counts) == 0:
                return None
            else:
                kmer_counts = convert_counts_to_dataframe(kmer_counts, ID)
                kmer_count_list.append(kmer_counts)
        return kmer_count_list
    except Exception as err:
        logging.exception(f'Error processing {os.path.basename(filepath)}')



def process_batch_result(kmer_results:list, k_pos):
    """
    Processes batch result from multiprocessing
    Loops through results for files in batch and retrieves the dataframes in the same 
    position as the kmer

    :param kmer_results: List of lists of dataframes for each ID. Each list maps position of given kmer
    :param k_pos: Position of kmer to extract from the list of dataframes
    :returns a dataframe for each kmer
    """
    # returns a list of list of kmer dataframe for each sample
    # get k position
    samples_k_results = pd.DataFrame(dtype=np.uint32)
    for sample_kmer in kmer_results:
        if sample_kmer is None:
            continue
        # merge kmers for each sample (dim = kmer x sample) along kmer
        samples_k_results = pd.concat([samples_k_results, sample_kmer[k_pos]], axis=1)
    # fillna with 0, move kmer as column and drop duplicates
    samples_k_results = (
        samples_k_results.fillna(0)
        .astype(np.uint32)
        # drop duplicates incase there are any
        .reset_index(names='kmer')
        .drop_duplicates()
    )
    return samples_k_results.set_index('kmer')


def merge_and_save_batch_kmer_files(folder_path, k, **kwargs):
    """
    Merges and saves batch files of kmer dataframes saved in a temporary folder
    :param folder_path: Path to saved batch files
    :param k: List of k-mers
    """
    data_folder = kwargs.get('data_folder')
    os.makedirs(f'data', exist_ok=True)

    cutoff = kwargs.get('cutoff', 0.9)
    for i in k:
        batch_files = glob(f'{folder_path}/{data_folder}/*{i}kmer*.parquet') # get saved batch files
        if len(batch_files) == 0:
            raise FileNotFoundError(f'{folder_path}/{data_folder} not found')
        
        kmer_files = []
        for r in range(len(batch_files)):
            temp = pd.read_parquet(batch_files[r])

            # drop singleton kmers (kmers with one count in at least 85% of the samples)
            # drop kmers that do not appear in cutoff (percentage of samples)
            # perc_singleton_or_zeros = np.mean((temp == 0) | (temp == 1), axis=1) # along samples (columns)
            # temp = temp.loc[perc_singleton_or_zeros < cutoff]
            kmer_files.append(temp)
            del temp
            
        kmer_files = pd.concat(kmer_files, axis=1).fillna(0).astype(np.uint32)
        kmer_files.to_parquet(f'data/{data_folder}_{i}kmer_prac.parquet')
        del kmer_files
    gc.collect()


def parallel_preprocessing2(file_paths, k, batch_size=120, max_workers=4, data_folder='train', **kwargs):
    """Preprocesses file in parallel"""
    temp_file = 'data/temp' 
    os.makedirs(temp_file, exist_ok=True)
    num_files = len(file_paths)
    
    print(f'A total of {num_files:,} files would be preprocessed in batches of {batch_size:,}\n')
    num_batches = (num_files + batch_size - 1) // batch_size
    
    for j, batch in enumerate(tqdm(get_batch(file_paths, batch_size=batch_size), total=num_batches)):
        # returns dictionary of k with their dataframes
        batch_kmer_results = Parallel(n_jobs=max_workers, backend="loky")(
            delayed(preprocess_fastq2)(file_path, k) for file_path in batch)
        if not isinstance(k, (list, tuple)):
            k = [k]
        for k_pos, kmer in enumerate(k):
            batch_sample_k_results = process_batch_result(batch_kmer_results, k_pos)
            filepath = os.path.join(temp_file, data_folder)
            os.makedirs(filepath, exist_ok=True)
            batch_sample_k_results.to_parquet(f'{filepath}/batch{str(j).zfill(2)}_sample_{kmer}kmer.parquet') 
            del batch_sample_k_results
        del batch_kmer_results
        gc.collect() 
    
    cutoff = kwargs.get('cutoff', 0.9)
    merge_and_save_batch_kmer_files(temp_file, k, data_folder=data_folder, cutoff=cutoff)

    # delete temp file
    if os.path.exists(temp_file+f'/{data_folder}'):
        shutil.rmtree(temp_file+f'/{data_folder}')
    shutil.rmtree(temp_file)


# for train and test
parallel_preprocessing2(test_files, [8], batch_size=100, data_folder='test', cutoff=0.9)

parallel_preprocessing2(train_files, [8], batch_size=100, data_folder='train', cutoff=0.9)


