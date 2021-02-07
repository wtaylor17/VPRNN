from vprnn.imdb_data import stash_imdb, stash_embeddings
import sys


if sys.argv[1:]:
    n = int(sys.argv[1])
else:
    n = 25000

stash_imdb(num_words=n)
stash_embeddings()
