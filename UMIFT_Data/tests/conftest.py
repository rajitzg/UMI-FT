def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs real datasets/checkpoints on this machine")
