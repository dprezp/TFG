import matplotlib.pyplot as plt
from aif360.datasets import AdultDataset, CompasDataset, GermanDataset, BankDataset
from aif360.datasets import BinaryLabelDataset
import numpy as np

# Dataset → atributo protegido
datasets_info = [
    ('Adult', AdultDataset, ['sex', 'race']),
    ('COMPAS', CompasDataset, ['sex', 'race']),
    ('German', GermanDataset, ['sex', 'age']),
    ('Bank', BankDataset, ['age'])
]

for name, DatasetClass, protected_attrs in datasets_info:
    dataset = DatasetClass()

    for attr in protected_attrs:
        dataset.protected_attribute_names([attr])
        values, counts = np.unique(dataset.features[:, dataset.protected_attribute_names.index(attr)],
                                   return_counts=True)

        # Preparar etiquetas
        labels = [f'{attr} = {int(val)}' for val in values]
        sizes = counts

        # Gráfico
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107'])
        ax.axis('equal')  # Círculo perfecto
        plt.title(f'{name} - Distribución de {attr}')
        plt.tight_layout()
        plt.savefig(f'pie_{name.lower()}_{attr}.png')
        plt.close()
