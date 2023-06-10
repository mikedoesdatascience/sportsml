import io
import pickle
import pathlib

from ..models.gnn import GraphModel


def save_graph_model(collection, checkpoint, metadata):
    model_bytes = pathlib.Path(checkpoint).read_bytes()
    doc = metadata.copy()
    doc["model"] = model_bytes
    collection.insert_one(doc)


def save_random_forest(collection, model, metadata):
    doc = metadata.copy()
    doc["model"] = pickle.dumps(model)
    collection.insert_one(doc)


def load_graph_model(document, model_class=GraphModel):
    model_bytes = io.BytesIO(document["model"])
    return model_class.load_from_checkpoint(model_bytes)


def load_random_forest(document):
    return pickle.loads(document["model"])
