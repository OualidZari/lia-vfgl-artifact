from torch_geometric.datasets import Planetoid, Twitch, Amazon

def download_datasets():
    datasets = [
        ('Cora', Planetoid, {'name': 'Cora'}),
        ('Citeseer', Planetoid, {'name': 'Citeseer'}),
        ('amazon_computer', Amazon, {'name': 'Computers'}),
        ('amazon_photo', Amazon, {'name': 'Photo'}),
        ('Twitch-DE', Twitch, {'name': 'DE'}),
        ('Twitch-EN', Twitch, {'name': 'EN'}),
        ('Twitch-ES', Twitch, {'name': 'ES'}),
        ('Twitch-FR', Twitch, {'name': 'FR'})
    ]

    for dataset_name, dataset_class, dataset_args in datasets:
        print(f"Downloading {dataset_name}...")
        dataset_class(root=f'datasets/{dataset_name}', **dataset_args)
        print(f"{dataset_name} downloaded successfully.")

if __name__ == "__main__":
    download_datasets()
