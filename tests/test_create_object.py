from BibVisualizationExpress.BibVisualizationExpress import BibVisualizationExpress


def test_object_creation():
    # Create an object from your library
    bib_express_viz = BibVisualizationExpress()

    # Assert that the object is not None
    assert bib_express_viz is not None
