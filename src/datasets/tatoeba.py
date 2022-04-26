# Internal imports
import nmslib

from ..util import DATA_PATH, download_url

# External imports
import os
import functools
import pandas as pd
import sentence_transformers
import h5py

TATOEBA_EN_URL = "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2"


def load_or_download_tsv(bottom_length_percentile, source_local_path, source_url, top_length_percentile):
	read_csv = functools.partial(pd.read_csv, filepath_or_buffer=source_local_path, sep='\t', header=None,
								 usecols=(2,), names=("text",))
	try:
		df = read_csv()
	except FileNotFoundError as e:  # No TSV file yet
		download_url(source_url, source_local_path)
		df = read_csv()
	df = df[~df['text'].str.contains('\n')]
	df.sort_values(by="text", key=lambda x: x.str.len(), inplace=True, ignore_index=True)
	df = df.iloc[round(bottom_length_percentile * len(df)):round(top_length_percentile * len(df))]
	return df


class TatoebaEN:
	def __init__(self, top_length_percentile=0.95, bottom_length_percentile=0.05, source_url=TATOEBA_EN_URL,
				 vectorization_method="transformer", device=None):
		file_name = os.path.basename(source_url)
		source_local_path = os.path.realpath(os.path.join(DATA_PATH, file_name))
		vectors_local_path = os.path.realpath(os.path.join(
			DATA_PATH, f"{file_name.split('.')[0]}_{vectorization_method}.h5"
		))
		index_local_path = os.path.realpath(os.path.join(
			DATA_PATH, f"{file_name.split('.')[0]}_{vectorization_method}.nms"
		))

		self.sentences = load_or_download_tsv(
			bottom_length_percentile, source_local_path, source_url,
			top_length_percentile
		)['text'].to_list()

		index: nmslib.dist.FloatIndex
		index = nmslib.init(data_type=nmslib.DataType.DENSE_VECTOR)
		try:
			index.loadIndex(index_local_path)
		except RuntimeError as e:
			vectors = self.get_vectors(device, vectors_local_path)
			index.addDataPointBatch(vectors)
			index.createIndex(print_progress=True)
			index.saveIndex(index_local_path)
		self.index = index

	def get_vectors(self, device, vectors_local_path, batch_size=128):
		try:
			with h5py.File(vectors_local_path, 'r') as f:
				vectors = f['vectors'][:]
		except FileNotFoundError as e:  # No vectors yet
			vectorizer = sentence_transformers.SentenceTransformer("all-mpnet-base-v2", device=device,
																   cache_folder=DATA_PATH)
			vectors = vectorizer.encode(self.sentences, batch_size=batch_size, show_progress_bar=True)
			with h5py.File(vectors_local_path, 'w') as f:
				f.create_dataset('vectors', data=vectors)
		return vectors

	# i = 0
			# num_batches = len(self.df) // batch_size
			# remainder = len(self.df) % batch_size
			# for i in tqdm.tqdm(range(num_batches)):
			# 	cur_slice = slice(i*batch_size, (i+1)*batch_size)
			# 	# do stuff
			# 	batch = vectorizer.en
			# 	pass
			# cur_slice = slice(len(self.df) - remainder, len(self.df))
