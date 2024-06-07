# first line: 13
@memory.cache
def cached_fit_transform(vectorizer, docs):
    return vectorizer.fit_transform(docs)
