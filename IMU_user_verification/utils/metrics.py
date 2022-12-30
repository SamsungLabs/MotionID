import numpy as np

from scipy.spatial.distance import pdist, squareform
from multiprocessing import Pool, cpu_count


def ncpus_from_njobs(njobs):
    return max(1, njobs) if njobs >= 0 else max(1, cpu_count() + njobs + 1)


def bootstrap_metric_thr_wrapper(args):
    ppdist_st, pdist_labels, thr, m, iters = args
    return bootstrap_metric_thr_far(ppdist_st, pdist_labels, thr, m=m, iters=iters)


def bootstrap_metric_thr_far(ppdist_st, pdist_labels, thr, m=200, iters=2000):
    if np.isscalar(thr):
        thr = np.array([thr])
    unique_labels = np.sort(np.unique(pdist_labels))
    n = len(unique_labels)
    labels2indices = {
        label: np.where(pdist_labels == label)[0]
        for label in unique_labels
    }
    labels2nonindices = {
        label: np.where(pdist_labels != label)[0]
        for label in unique_labels
    }
    fars = []
    for iter_num in np.arange(iters):#tqdm.tqdm_notebook(np.arange(iters)):
        sampled_labels = np.random.choice(unique_labels, size=n, replace=True)
        sampled_enroll_indices_per_label = np.stack([
            np.repeat(np.random.choice(labels2nonindices[label], size=n-1, replace=True), m)
            for label in sampled_labels
        ], axis=0)
        sampled_verify_indices_per_label = np.stack([
            np.tile(np.random.choice(labels2indices[label], size=m, replace=True), n-1)
            for label in sampled_labels
        ], axis=0)
        enroll_raws = sampled_enroll_indices_per_label.reshape(-1)
        verify_raws = sampled_verify_indices_per_label.reshape(-1)
        match_scores = ppdist_st[enroll_raws, verify_raws].copy()
        sample_far = np.array([
            np.mean(match_scores >= thr_el)
            for thr_el in thr
        ])
        #sns.distplot(match_scores.ravel())
        fars.append(sample_far)
    return np.stack(fars, axis=0)


def bootstrap_metric_thr_frr_wrapper(args):
    ppdist_st, pdist_labels, thr, m, iters = args
    return bootstrap_metric_thr_frr(ppdist_st, pdist_labels, thr, m=m, iters=iters)


def bootstrap_metric_thr_frr(ppdist_st, pdist_labels, thr, m=200, iters=1000):
    if np.isscalar(thr):
        thr = np.array([thr])
    unique_labels = np.sort(np.unique(pdist_labels))
    n = len(unique_labels)
    labels2indices = {
        label: np.where(pdist_labels == label)[0]
        for label in unique_labels
    }
    labels2nonindices = {
        label: np.where(pdist_labels != label)[0]
        for label in unique_labels
    }
    frrs = []
    for iter_num in np.arange(iters):#tqdm.tqdm_notebook(np.arange(iters)):
        sampled_labels = np.random.choice(unique_labels, size=n, replace=True)
        sampled_verify_indices = np.array([np.random.choice(labels2indices[label]) for label in sampled_labels])
        sampled_verify_indices_per_label = np.stack([
            np.random.choice(labels2indices[label], size=m, replace=True)
            for label in sampled_labels
        ], axis=0)
        match_scores = [
            ppdist_st[sampled_label, sampled_verify_indices].copy()
            for sampled_label, sampled_verify_indices in zip(sampled_verify_indices, sampled_verify_indices_per_label)
        ]
        #sns.distplot(np.stack(match_scores, 0).reshape(-1))
        sample_frr = np.array([
            
            np.mean(np.stack(match_scores, 0).reshape(-1) <= thr_el)
            for thr_el in thr
        ])
        frrs.append(sample_frr)
    return np.stack(frrs, axis=0)


def parallel_bootstrap_metric_thr(ppdist_st, pdist_labels, thr, m=200, iters=1000, njobs=40):
    ncpus = ncpus_from_njobs(njobs)
    
    chunk_size = iters // njobs
    last_iter_size = iters % njobs
    if last_iter_size:
        chunk_size += 1
    high_iters = chunk_size * njobs
    
    chunk_iters = [chunk_size] * njobs
    ppdist_st_list = [ppdist_st] * njobs
    pdist_labels_list = [pdist_labels] * njobs
    thr_list = [thr] * njobs
    mm = [m] * njobs
    
    pool_args = zip(ppdist_st_list, pdist_labels_list, thr_list, mm, chunk_iters)
    
    with Pool(ncpus) as pool:
        res = np.concatenate(pool.map(bootstrap_metric_thr_wrapper, pool_args), axis=0)
        print(res.shape)
        return res[:iters]


def parallel_bootstrap_metric_thr_frr(ppdist_st, pdist_labels, thr, m=200, iters=1000, njobs=40):
    ncpus = ncpus_from_njobs(njobs)
    
    chunk_size = iters // njobs
    last_iter_size = iters % njobs
    if last_iter_size:
        chunk_size += 1
    high_iters = chunk_size * njobs
    
    chunk_iters = [chunk_size] * njobs
    ppdist_st_list = [ppdist_st] * njobs
    pdist_labels_list = [pdist_labels] * njobs
    thr_list = [thr] * njobs
    mm = [m] * njobs
    
    pool_args = zip(ppdist_st_list, pdist_labels_list, thr_list, mm, chunk_iters)
    
    with Pool(ncpus) as pool:
        res = np.concatenate(pool.map(bootstrap_metric_thr_frr_wrapper, pool_args), axis=0)
        print(res.shape)
        return res[:iters]


def compute_frr10_rough(scores, class_labels, thr_space, iters=100, njobs=10):
    fars = bootstrap_metric_thr_far(scores, class_labels, thr_space, iters=iters)
    frrs = bootstrap_metric_thr_frr(scores, class_labels, thr_space, iters=iters)
    
    frr_means = frrs.mean(0)
    far_means = fars.mean(0)
    
    ind_to = np.where(frr_means > 0.1)[0][0]
    return frr_means[ind_to - 1], far_means[ind_to - 1]


def est_frr_q(scores, class_labels, iters=1000, m=90, q=0.1, p=0.5):
    def est_frr(thr):
        frr_distribution = bootstrap_metric_thr_frr(
            scores,
            class_labels,
            np.linspace(thr, thr, 1),
            m=m,
            iters=iters,
        )
        return np.quantile(frr_distribution, p)
    def est_far(thr):
        far_distribution = bootstrap_metric_thr_far(
            scores,
            class_labels,
            np.linspace(thr, thr, 1),
            m=m,
            iters=iters,
        )
        return np.quantile(far_distribution, p)

    left, right = 0.0, 1.0
    #frr_left = bootstrap_metric_thr_frr(scores, class_labels, np.linspace(left, left, 1), m=m, iters=iters).mean()
    #frr_right = bootstrap_metric_thr_frr(scores, class_labels, np.linspace(right, right, 1), m=m, iters=iters).mean()
    frr_left = est_frr(left)
    frr_right = est_frr(right)
    if frr_left >= q:
        # oops, thr = 0 already frr >= q, nothing we can do here
        #return (bootstrap_metric_thr_far(scores, class_labels, np.linspace(left, left, 1), m=m, iters=iters).mean(), frr_left)
        return est_far(left), frr_left
    if frr_right <= q:
        # oops, thr = 1.0 already frr <= q, nothing we can do here
        #return (bootstrap_metric_thr_far(scores, class_labels, np.linspace(right, right, 1), m=m, iters=iters).mean(), frr_right)
        return est_far(right), frr_right

    while True:#np.abs(right - left) < 1e-6 or (np.abs(frr_right - q) < 1e-6 and np.abs(frr_left - q) < 1e-6):
        if frr_left > q:
            left -= (right - left)
            left = max(0.0, left)
            #frr_left = bootstrap_metric_thr_frr(scores, class_labels, np.linspace(left, left, 1), m=m, iters=iters).mean()
            frr_left = est_frr(left)
            continue
        if frr_right < q:
            right += (right - left)
            right = min(1.0, right)
            #frr_right = bootstrap_metric_thr_frr(scores, class_labels, np.linspace(right, right, 1), m=m, iters=iters).mean()
            frr_right = est_frr(right)

        mid = (right + left) / 2
        #frr_mid = bootstrap_metric_thr_frr(scores, class_labels, np.linspace(mid, mid, 1), m=m, iters=iters).mean()
        frr_mid = est_frr(mid)
        #print(mid, frr_mid * 100)
        if np.abs(frr_mid - q) < 5e-4:
            # found that sweet frr
            #return (bootstrap_metric_thr_far(scores, class_labels, np.linspace(mid, mid, 1), m=m, iters=iters).mean(), frr_mid)
            return est_far(mid), frr_mid
        elif frr_mid < q:
            # mid is now new left
            left, frr_left = mid, frr_mid
        elif frr_mid > q:
            # mid is now new right
            right, frr_right = mid, frr_mid
        else:
            # WTF
            print(f'what? {left}[{frr_left}], {mid}[{frr_mid}], {right}[{frr_right}]')
    return


def binary_bootstrap_metric_thr_far(ppdist_st, pdist_labels, thr, m=200, iters=2000):
    if np.isscalar(thr):
        thr = np.array([thr])
    genuine_indices = np.where(pdist_labels == 1)[0]
    impostor_indices = np.where(pdist_labels == 0)[0]

    # bootstrap
    # first, sample m genuine samples with replacement
    # then, sample m impostor samples with replacement
    fars = []
    for it in range(iters):
        genuine_indices_choice = np.random.choice(genuine_indices, size=m)
        impostor_indices_choice = np.random.choice(impostor_indices, size=m)

        match_scores = ppdist_st[genuine_indices_choice, impostor_indices_choice].copy()
        #match_scores = match_scores.reshape((iters, m))
        #sample_far = np.stack([
        #    np.mean(match_scores >= thr_el)
        #    for thr_el in thr
        #])
        #fars.append(sample_far)
        #return np.stack(fars, axis=0)
        #fars = np.stack([
        #    np.mean(match_scores >= thr_el, axis=-1)
        #    for thr_el in thr
        #], axis=1)
        sample_far = np.stack([
            np.mean(match_scores >= thr_el)
            for thr_el in thr
        ])
        fars.append(sample_far)

    #return fars
    return np.stack(fars, axis=0)

def binary_bootstrap_metric_thr_frr(ppdist_st, pdist_labels, thr, m=200, iters=2000):
    if np.isscalar(thr):
        thr = np.array([thr])
    genuine_indices = np.where(pdist_labels == 1)[0]
    verify_indices = np.where(pdist_labels == 1)[0]

    # bootstrap
    # first, sample m genuine samples with replacement
    # then, sample m impostor samples with replacement
    frrs = []
    for it in range(iters):
        genuine_indices_choice = np.random.choice(genuine_indices, size=m)
        verify_indices_choice = np.random.choice(verify_indices, size=m)

        match_scores = ppdist_st[genuine_indices_choice, verify_indices_choice].copy()
        #match_scores = match_scores.reshape((iters, m))
        #sample_frr = np.array([
        #    np.mean(match_scores <= thr_el)
        #    for thr_el in thr
        #])
        #frrs.append(sample_frr)
        #return np.stack(frrs, axis=0)
        #frrs = np.stack([
        #    np.mean(match_scores <= thr_el, axis=-1)
        #    for thr_el in thr
        #], axis=1)
        #return frrs
        sample_frr = np.array([
            np.mean(match_scores <= thr_el)
            for thr_el in thr
        ])
        frrs.append(sample_frr)
    return np.stack(frrs, axis=0)


def binary_est_frr_q(scores, class_labels, iters=1000, m=90, q=0.1, p=0.5):
    def est_frr(thr):
        frr_distribution = binary_bootstrap_metric_thr_frr(
            scores,
            class_labels,
            np.linspace(thr, thr, 1),
            m=m,
            iters=iters,
        )[..., 0]
        return np.quantile(frr_distribution, p)
    def est_far(thr):
        far_distribution = binary_bootstrap_metric_thr_far(
            scores,
            class_labels,
            np.linspace(thr, thr, 1),
            m=m,
            iters=iters,
        )[..., 0]
        return np.quantile(far_distribution, p)

    left, right = 0.0, 1.0
    frr_left = est_frr(left)
    frr_right = est_frr(right)
    if frr_left >= q:
        # oops, thr = 0 already frr >= q, nothing we can do here
        print(f'frr_left >= q, oops')
        return est_far(left), frr_left
    if frr_right <= q:
        # oops, thr = 1.0 already frr <= q, nothing we can do here
        print(f'frr_right <= q, oops')
        #return (bootstrap_metric_thr_far(scores, class_labels, np.linspace(right, right, 1), m=m, iters=iters).mean(), frr_right)
        return est_far(right), frr_right

    nflag = 0

    while True:#np.abs(right - left) < 1e-6 or (np.abs(frr_right - q) < 1e-6 and np.abs(frr_left - q) < 1e-6):
        if frr_left > q:
            if nflag >= 6:
                return est_far(left), frr_left
            left -= (right - left)
            left = max(0.0, left)
            frr_left = est_frr(left)
            nflag += 1
            continue
        if frr_right < q:
            if nflag >= 6:
                return est_far(right), frr_right
            right += (right - left)
            right = min(1.0, right)
            frr_right = est_frr(right)
            nflag += 1
            continue

        mid = (right + left) / 2
        frr_mid = est_frr(mid)
        #print(left, mid, right)
        #print(frr_left, frr_mid, frr_right, nflag)
        #print('-'*80)
        if np.abs(right - left) < 1e-9:
            return est_far(left), frr_left
        if np.abs(frr_mid - q) < 5e-4:
            # found that sweet frr
            return est_far(mid), frr_mid
        elif frr_mid < q:
            # mid is now new left
            left, frr_left = mid, frr_mid
        elif frr_mid > q:
            # mid is now new right
            right, frr_right = mid, frr_mid
        else:
            # WTF
            print(f'what? {left}[{frr_left}], {mid}[{frr_mid}], {right}[{frr_right}]')
    return