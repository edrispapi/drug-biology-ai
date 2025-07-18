from Bio import SeqIO

def load_fasta(filepath):
    """بارگذاری و استخراج توالی پروتئین از فایل FASTA"""
    record = SeqIO.read(filepath, "fasta")
    return str(record.seq)
