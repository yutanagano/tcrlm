from src.datahandling import samplers


class TestSortedBatchSampler:
    def test_iter(self):
        sampler = samplers.SortedBatchSampler(
            num_samples=1000,
            batch_size=5,
            sort_a=(lambda x: 1000 - x)
        )

        for i, batch in enumerate(sampler):
            if i < 100:
                expected = list(reversed(range(495-5*i,500-5*i)))
                assert batch == expected
                continue
            
            expected = list(reversed(range(995-5*(i-100),1000-5*(i-100))))
            assert batch == expected


    def test_iter_shuffle(self):
        sampler = samplers.SortedBatchSampler(
            num_samples=1000,
            batch_size=5,
            sort_a=(lambda x: x),
            shuffle=True
        )

        for i, batch in enumerate(sampler):
            if i == 0:
                # assert first list of indices is not [0,1,2,3,4] as the
                # probability of this happening with shuffle is negligible
                assert batch != [0,1,2,3,4]

            # assert that the list of indices is always ascending
            assert batch == sorted(batch)


    def test_len(self):
        sampler = samplers.SortedBatchSampler(
            num_samples=1000,
            batch_size=5,
            sort_a=(lambda x: x)
        )

        assert len(sampler) == 200