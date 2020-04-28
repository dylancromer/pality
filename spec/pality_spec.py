import pytest
import numpy as np
import pality


def describe_Pca():

    @pytest.fixture
    def test_data():
        xs = np.random.rand(50)
        ys = np.random.rand(*xs.shape) + 0.6*xs
        data = np.stack((xs, ys)).T
        data = data - data.mean(axis=0)[None, :]
        return data / data.std(axis=0)[None, :]

    def it_constructs_the_pcs_of_some_data(test_data):
        pca = pality.Pca.calculate(test_data)
        assert np.allclose(np.abs(pca.weights), np.array([[0.5, 0.5], [0.5, 0.5]]))

    def describe_svd():

        def it_decomposes_a_matrix_into_svd_form():
            matrix = np.array([[1, 2], [3, 4]])

            u, s, v = pality.Pca.svd(matrix)

            u_should_be = np.array([[0.404554, 0.914514], [0.914514, -0.404554]])
            s_should_be = np.array([5.46499, 0.365966])
            v_should_be = np.array([[0.576048, 0.817416] , [-0.817416, 0.576048]])

            assert np.allclose(u, u_should_be)
            assert np.allclose(s, s_should_be)
            assert np.allclose(v, v_should_be)

    def describe_basis_vecs_from_svd():

        def it_extracts_the_basis_vectors_from_the_svd_matrices():
            u = np.array([[1, 2], [3, 4], [5, 6]])
            s = np.array([1, 2])

            basis_vecs = pality.Pca.basis_vecs_from_svd(u, s)

            assert np.all(basis_vecs == np.array([[1, 4], [3, 8], [5, 12]]) / np.sqrt(2))

    def describe_weights_from_svd():

        def it_extracts_the_weights_from_v():
            v = np.random.rand(10, 10)

            weights = pality.Pca.weights_from_svd(v)

            assert np.all(weights == v/np.sqrt(10))

    def describe_explained_var_from_s():

        def it_returns_the_fractions_of_variance_explained_by_each_pc():
            s = np.ones(10)
            expl_var = pality.Pca.explained_var_from_s(s, 3)
            assert np.all(expl_var == 0.5 / (10*0.5))
