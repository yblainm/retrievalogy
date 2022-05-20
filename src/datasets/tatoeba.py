# Internal imports
from typing import List


from ..util import DATA_PATH, download_url

# External imports
import os
import functools
import pandas as pd
import nmslib
import h5py
import sentence_transformers
import fasttext
from numba import jit, prange
from numpy.typing import ArrayLike
# from pandas import DataFrame
from numba_progress import ProgressBar

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


# @jit(forceobj=True, parallel=True)
def get_fasttext(vectorizer: fasttext.FastText, sentences: List[str], out_array: ArrayLike, progress_bar: ProgressBar = None):
	# for i in tqdm(range(len(sentences))):
	for i in range(len(sentences)):
		out_array[i] = vectorizer.get_sentence_vector(sentences[i])
		if progress_bar:
			progress_bar.update(1)
	return out_array

class TatoebaEN:
	def __init__(self, top_length_percentile=0.95, bottom_length_percentile=0.05, source_url=TATOEBA_EN_URL,
				 vectorization_method="transformer", device=None):
		assert vectorization_method in ("transformer", "fasttext")
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
		index = nmslib.init(
			data_type=nmslib.DataType.DENSE_VECTOR,
			space='cosinesimil' if vectorization_method == "transformer" else 'l2'
		)
		try:
			index.loadIndex(index_local_path)
		except RuntimeError as e:
			vectors = self.get_vectors(device, vectors_local_path, vectorization_method=vectorization_method)
			index.addDataPointBatch(vectors)
			index.createIndex(print_progress=True)
			index.saveIndex(index_local_path)
		self.index = index

	def get_vectors(self, device, vectors_local_path, batch_size=128, vectorization_method="transformer"):
		try:
			with h5py.File(vectors_local_path, 'r') as f:
				vectors = f['vectors'][:]
		except FileNotFoundError as e:  # No vectors yet
			if vectorization_method == "transformer":
				vectorizer = sentence_transformers.SentenceTransformer("all-mpnet-base-v2", device=device,
																	   cache_folder=DATA_PATH)
				vectors = vectorizer.encode(self.sentences, batch_size=batch_size, show_progress_bar=True)
				with h5py.File(vectors_local_path, 'w') as f:
					f.create_dataset('vectors', data=vectors)
			elif vectorization_method == "fasttext":
				vectorizer = fasttext.load_model(os.path.realpath(os.path.join(
					DATA_PATH, "cc.en.300.bin"
				)))
				with h5py.File(vectors_local_path, 'w') as f:
					f.create_dataset('vectors', shape=(len(self.sentences), 300))

					with ProgressBar(total=len(self.sentences)) as progress:
						vectors = get_fasttext(vectorizer, self.sentences, f['vectors'], progress_bar=progress)
						del vectorizer

					vectors = vectors[:]
		return vectors
