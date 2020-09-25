import unittest
import numpy as np
import pandas as pd
import DFInfo

class TestCalc(unittest.TestCase):

  def test_inspect(self):
    # get devuelve un df con información de interés sobre cada feature

    # Construir un DataFrame con tres columnas y dos filas, valores nulos, etc
    fakeCols = ['A', 'B', 'C']
    fakeData = {
      'row_1': [2, 1], 
      'row_2': [3, 'f', 'h'],
      'row_3': [4, 'f', 'h']
    }
    dfX = pd.DataFrame.from_dict(fakeData, orient='index', columns=fakeCols)  
    dfX = dfX.astype(dtype= {"A":"int64"})
    dfX['target'] = dfX['A'] * 3.14

    # Act
    r = DFInfo.inspect(dfX, 'target')

    # Assertions

    # Tiene la estructura esperada
    variables = ['count', 'dtype', 'nulls', 'unique']
    columnas = ['feature', 'variable', 'value']
    self.assertEqual(r.shape, (len(fakeCols) * len(variables), len(columnas)), 'Tiene la estructura esperada')

    # Tiene las columnas esperadas
    np.testing.assert_array_equal(r.columns.values, columnas, 'No se han recibido las columnas esperadas')
    # Están las series esperadas
    table = r.pivot(values='value', index='feature', columns='variable')    
    np.testing.assert_array_equal(table.columns.values, variables)

    # nulls: La columna C tiene un valor nulo => 50%    
    f = (r['variable'] == 'nulls') & (r['value'] != 0)
    actual = r.loc[f][['feature', 'value']].set_index('feature')['value']
    self.assertEqual(actual.to_dict(), {'C': 0.5}, 'La columna tiene un valor nulo')

    # count: El resto son no nulos
    f = (r['variable'] == 'count')
    actual = r.loc[f][['value']].sum().values
    self.assertEqual(actual, 8, 'El resto no son nulos')

    # unique
    f = (r['variable'] == 'unique')
    actual = r.loc[f][['value']].sum().values
    self.assertEqual(actual, 6, 'Valores únicos, incluyendo los numéricos')


# End of tests

if __name__ == '__main__':
    unittest.main()
