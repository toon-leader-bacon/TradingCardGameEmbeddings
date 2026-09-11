import torch

from src.dojos.dojo import Split
from src.dojos.v2.NocabWorkspace import NocabDojo
from src.encoder_model.single_card_model import SingleCardModel


class DemoTrainingLoop:
    def __init__(self):
        pass

    def example_training_usage(self):
        model: SingleCardModel = SingleCardModel()  # example
        # In the future there may be many multiple dojos per training
        # loop, so this should be a list.
        # But it's not clear how to handle multiple dojos per loop. Some are
        # much larger than others, some have a richer signal-to-noise ratio than others.
        # some are easier than others etc.
        dojo = NocabDojo()  # example
        dojo.prepare_splits(shuffle=True, rng_seed=42)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # example

        # Train the model
        best_test_loss = float("inf")
        for epoch in range(10):
            dojo.reset_split(split=Split.TRAIN, shuffle=True)
            for batch in dojo.next_batch(Split.TRAIN, batch_size=128):
                # Pass the inputs to the model to generate embeddings
                # Provide those embeddings to the dojo to generate the loss
                embeddings = model.forward(batch.inputs)
                loss = dojo.compute_loss(embeddings, batch.labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Test the model
            total_test_loss = 0.0
            total_test_samples = 0
            for inputs, labels in dojo.next_batch(Split.TEST, batch_size=128):
                # TODO: Think about how to signal/ produce the full batch without
                # looping over the split/ duplicating elements in the split.
                with torch.no_grad():
                    embeddings = model.forward(inputs)
                    loss = dojo.compute_loss(embeddings, labels)
                total_test_loss += loss.item()
                total_test_samples += len(inputs)
            print(f"Average Test loss: {total_test_loss / total_test_samples}")
            if total_test_loss < best_test_loss:
                best_test_loss = total_test_loss
                # Save the model
        # End of all epochs

        # validate the model
        total_validation_loss = 0.0
        total_validation_samples = 0
        for inputs, labels in dojo.next_batch(Split.VALIDATION, batch_size=128):
            with torch.no_grad():
                embeddings = model.forward(inputs)
                loss = dojo.compute_loss(embeddings, labels)
            total_validation_loss += loss.item()
            total_validation_samples += len(inputs)
        print(
            f"Average Validation loss: {total_validation_loss / total_validation_samples}"
        )
