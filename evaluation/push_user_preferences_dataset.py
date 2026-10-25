import os
import uuid

import dotenv
from langsmith import Client

dotenv.load_dotenv()

DATASET_NAME = os.environ.get("LANGSMITH_USER_PREFERENCES_DATASET", "news-user-preferences")

# Fixed namespace so every profile always gets the same example id (a uuid5 of its text),
# which lets a re-run overwrite the existing rows instead of duplicating them.
_ID_NAMESPACE = uuid.UUID("212d21e0-64dc-4a02-90ea-2876dee69fa5")

_PROFILES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "user_profiles.txt")
with open(_PROFILES_PATH) as f:
    USER_PREFERENCES = [line.strip() for line in f if line.strip()]


def get_or_create_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        print("Yeah")
        return client.read_dataset(dataset_name=DATASET_NAME)
    return client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Sample user preference profiles for evaluating the news "
            "orchestrator's search-query generation across topics."
        ),
    )


def push_examples(client: Client, dataset_id) -> tuple[int, int, int]:
    """Make the dataset match user_profiles.txt: overwrite examples that already exist,
    add the new ones and delete the ones no longer in the file.
    Returns (created, updated, deleted).

    The create and update endpoints are separate: creating an id that already exists is a
    409 (also for ids deleted in the UI, which stay reserved), so existing ones are updated.
    """
    examples = {
        uuid.uuid5(_ID_NAMESPACE, preference): {"user_profile": preference}
        for preference in USER_PREFERENCES
    }
    existing_ids = {e.id for e in client.list_examples(dataset_id=dataset_id)}

    to_create = [{"id": i, "inputs": inputs} for i, inputs in examples.items() if i not in existing_ids]
    to_update = [{"id": i, "inputs": inputs} for i, inputs in examples.items() if i in existing_ids]
    stale_ids = [i for i in existing_ids if i not in examples]

    if to_create:
        client.create_examples(dataset_id=dataset_id, examples=to_create)
    if to_update:
        client.update_examples(dataset_id=dataset_id, updates=to_update)
    if stale_ids:
        client.delete_examples(stale_ids)
    return len(to_create), len(to_update), len(stale_ids)


def main() -> None:
    client = Client()
    dataset = get_or_create_dataset(client)
    created, updated, deleted = push_examples(client, dataset.id)
    print(
        f"Dataset '{DATASET_NAME}' ({dataset.id}): created {created}, overwrote {updated}, "
        f"deleted {deleted} stale example(s)."
    )


if __name__ == "__main__":
    main()
