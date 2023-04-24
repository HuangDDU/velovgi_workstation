import os
import pickle

import scanpy as sc

def write_adata(adata, dirname="tmp"):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        print("create %s" % dirname)
        if "sample_recover" in adata.uns.keys():
            # 需要单独保存uns中的sample_recover
            sample_recover_pkl_filename = "%s/sample_recover.pkl" % dirname
            with open(sample_recover_pkl_filename, "wb") as f:
                pickle.dump(adata.uns["sample_recover"], f)
            del adata.uns["sample_recover"]
            print("save %s" % sample_recover_pkl_filename)
        # 布尔值转化为数字，方便保存
        is_sampled_key = "is_sampled"
        if is_sampled_key in adata.obs.columns:
            adata.obs[is_sampled_key] = adata.obs[is_sampled_key].apply(lambda x: 0 if x else 1)
        # 保存
        adata_filename = "%s/adata.h5ad" % dirname
        adata.write(adata_filename)
        print("save %s" % adata_filename)
    else:
        print("%s exist!" % dirname)


def read_adata(dirname="tmp"):
    if os.path.exists(dirname):
        adata_filename = "%s/adata.h5ad" % dirname
        adata = sc.read_h5ad(adata_filename)
        print("load %s" % adata_filename)
        if "sample_recover.pkl" in os.listdir(dirname):
            # 需要单独读取uns中的sample_recover
            sample_recover_pkl_filename = "%s/sample_recover.pkl" % dirname
            with open(sample_recover_pkl_filename, "rb") as f:
                adata.uns["sample_recover"] = pickle.load(f)
            print("load %s" % sample_recover_pkl_filename)
        return adata
    else:
        print("%s not exist!" % dirname)
