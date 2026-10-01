"""Unified training — aligned with voices/train_disco_demographic.py."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from diadem.data_io import read_data
from diadem.models.diadem import DISCO_DemographicWeighted
from diadem.models.disco_baseline import DISCO
from diadem.registry import load_model_cfg
from diadem.utils.utils import save_object


def calc_stats_diadem(model, Xi, Yi, Ya, Y, A, I, batch_size=128, n_subset=1000,
                      eval_aggreg=True, all_annotator_metadata=None, all_annotator_ids=None):
    """Match voices/train_disco_demographic.calc_stats for demographic model."""
    drop_p = model.drop_p
    model.drop_p = 0.0
    ptrs = np.arange(Y.shape[0])
    if n_subset > 0 and len(ptrs) > n_subset:
        ptrs = np.random.permutation(len(ptrs))[:n_subset]

    L = KLi = KLa = acc = agg_acc = Ns = 0.0
    preds, targets = [], []
    use_id = getattr(model, "use_identity_embedding", False)

    for ptr_s in range(0, len(ptrs), batch_size):
        idx = ptrs[ptr_s : ptr_s + batch_size]
        xi = tf.cast(Xi[idx], tf.float32)
        a = tf.cast(A[idx, 1:], tf.float32)
        a_idx = tf.cast(A[idx, 0], tf.int32) if use_id else None
        y = tf.cast(Y[idx], tf.float32)
        yi = tf.cast(Yi[idx], tf.float32)
        ya = tf.cast(Ya[idx], tf.float32)
        I_s = tf.constant(I[idx].flatten().astype(np.int32))
        y_ind = tf.argmax(y, 1)

        z, z_a = model.encode(xi, a, a_idx=a_idx)
        pY, _ = model.decode_y(z)
        pYi, _ = model.decode_yi(z, z_a)
        pYa, _ = model.decode_ya(z)
        L_t, _, KLi_t, KLa_t = model.calc_loss(I_s, y, yi, ya, pY, pYi, pYa)
        n = y.shape[0]
        L += float(L_t) * n
        KLi += float(KLi_t) * n
        KLa += float(KLa_t) * n

        y_pred = tf.argmax(pY, 1)
        acc += float(tf.reduce_sum(tf.cast(tf.equal(y_pred, y_ind), tf.float32)))
        preds.extend(y_pred.numpy().tolist())
        targets.extend(y_ind.numpy().tolist())

        if eval_aggreg and all_annotator_metadata is not None:
            for s in range(xi.shape[0]):
                xs = tf.expand_dims(xi[s], 0)
                py, _ = model.decode_y_ensemble(
                    xs,
                    all_annotator_metadata=all_annotator_metadata,
                    all_annotator_ids=all_annotator_ids,
                )
                y_mode = tf.argmax(tf.reduce_mean(py, 0))
                agg_acc += float(tf.cast(tf.equal(y_mode, y_ind[s]), tf.float32))

    model.drop_p = drop_p
    Ns = len(ptrs)
    acc = acc / Ns
    L, KLi, KLa = L / Ns, KLi / Ns, KLa / Ns
    if not (eval_aggreg and all_annotator_metadata is not None):
        agg_acc = acc
    else:
        agg_acc = agg_acc / Ns
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return acc, L, KLi, KLa, agg_acc, f1


def calc_stats_disco(model, data, batch_size=128, n_subset=1000):
    model.drop_p = 0.0
    ptrs = np.arange(data["Y"].shape[0])
    if n_subset > 0 and len(ptrs) > n_subset:
        ptrs = np.random.permutation(len(ptrs))[:n_subset]
    L = acc = Ns = 0.0
    preds, targets = [], []
    for ptr_s in range(0, len(ptrs), batch_size):
        idx = ptrs[ptr_s : ptr_s + batch_size]
        xi = tf.cast(data["Xi"][idx], tf.float32)
        a = tf.cast(data["A"][idx, :], tf.float32)
        y = tf.cast(data["Y"][idx], tf.float32)
        yi = tf.cast(data["Yi"][idx], tf.float32)
        ya = tf.cast(data["Ya"][idx], tf.float32)
        z = model.encode(xi, a)
        pY, _ = model.decode_y(z)
        pYi, _ = model.decode_yi(z)
        pYa, _ = model.decode_ya(z)
        L_t, _, _, _ = model.calc_loss(y, yi, ya, pY, pYi, pYa)
        L += float(L_t) * len(idx)
        y_ind = tf.argmax(y, 1)
        y_pred = tf.argmax(pY, 1)
        acc += float(tf.reduce_sum(tf.cast(tf.equal(y_pred, y_ind), tf.float32)))
        preds.extend(y_pred.numpy().tolist())
        targets.extend(y_ind.numpy().tolist())
        Ns += len(idx)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return acc / Ns, L / Ns, f1


def train_model(data_dir: Path, model_name: str, dataset: str = "voiced", gpu_id: int = 0):
    print(f"[train] model={model_name} dataset={dataset} gpu_id={gpu_id}")
    print(f"[train] data_dir={data_dir}")
    sys.stdout.flush()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id >= 0 else "-1"
    mp = load_model_cfg(model_name, dataset)
    print(f"[train] hyperparams: n_epoch={mp['n_epoch']} batch_size={mp['batch_size']} "
          f"act={mp['act_fx']} lr={mp['learning_rate']}")
    sys.stdout.flush()
    data = read_data(data_dir, model_name)
    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    if model_name == "diadem":
        use_id = bool(mp.get("use_identity_embedding", True))
        id_alpha_mode = str(mp.get("identity_alpha_mode", "additive")).lower().strip()
        print("[train] building DISCO_DemographicWeighted (DiADEM) ...")
        print(
            f"[train] use_identity_embedding={use_id} "
            f"identity_alpha_mode={id_alpha_mode} "
            f"n_annotators={data.get('n_annotators')}"
        )
        sys.stdout.flush()
        model = DISCO_DemographicWeighted(
            xi_dim=data["n_xi"],
            yi_dim=data["yi_dim"],
            ya_dim=data["ya_dim"],
            y_dim=data["y_dim"],
            a_dim=data["n_a"],
            lat_i_dim=mp["lat_i_dim"],
            lat_a_dim=mp["lat_a_dim"],
            lat_dim=mp["lat_dim"],
            act_fx=mp["act_fx"],
            init_type=mp["weight_init_scheme"],
            lat_fusion_type=mp["lat_fusion_type"],
            drop_p=mp["drop_p"],
            gamma_i=mp["gamma_i"],
            gamma_a=mp["gamma_a"],
            metadata_feature_names=data["metadata_feature_names"],
            metadata_feature_dims=data["metadata_feature_dims"],
            use_interaction_features=mp.get("use_interaction_features", True),
            interaction_dim=mp.get("interaction_dim", 64),
            use_disagreement_detection=mp.get("use_disagreement_detection", True),
            disagreement_lambda=mp.get("disagreement_lambda", 2.0),
            use_identity_embedding=use_id,
            identity_alpha_mode=id_alpha_mode,
            n_annotators=data.get("n_annotators"),
        )
        model.set_opt(mp["opt_type"], mp["learning_rate"])
        print("[train] DiADEM model built")
    else:
        print("[train] building DISCO baseline ...")
        sys.stdout.flush()
        model = DISCO(
            xi_dim=data["n_xi"],
            yi_dim=data["yi_dim"],
            ya_dim=data["ya_dim"],
            y_dim=data["y_dim"],
            a_dim=data["n_a"],
            lat_i_dim=mp["lat_i_dim"],
            lat_a_dim=mp["lat_a_dim"],
            lat_dim=mp["lat_dim"],
            act_fx=mp["act_fx"],
            init_type=mp["weight_init_scheme"],
            lat_fusion_type=mp["lat_fusion_type"],
            drop_p=mp["drop_p"],
            gamma_i=mp["gamma_i"],
            gamma_a=mp["gamma_a"],
        )
        model.set_opt(mp["opt_type"], mp["learning_rate"])
        print("[train] DisCo baseline model built")
    sys.stdout.flush()

    batch_size = mp["batch_size"]
    n_epoch = mp["n_epoch"]
    eval_every = mp.get("eval_every", 1)
    save_every = mp.get("save_every", 10)
    update_radius = mp.get("update_radius", -1.0)

    history = []
    print(f"[train] computing init stats (subset) ...")
    sys.stdout.flush()
    if model_name == "diadem":
        acc, L, _, _, agg_acc, f1 = calc_stats_diadem(
            model, data["Xi"], data["Yi"], data["Ya"], data["Y"], data["A"], data["I"],
            batch_size,
            all_annotator_metadata=data["annotator_metadata"],
            all_annotator_ids=data.get("annotator_id_list"),
        )
        print(f"[diadem] Init Acc={acc:.4f} Agg={agg_acc:.4f} L={L:.4f} F1={f1:.4f}")
    else:
        acc, L, f1 = calc_stats_disco(model, data, batch_size)
        print(f"[disco] Init Acc={acc:.4f} L={L:.4f} F1={f1:.4f}")
    sys.stdout.flush()

    print(f"[train] starting {n_epoch} epochs (batch_size={batch_size}, n_train={data['Y'].shape[0]})")
    sys.stdout.flush()
    for e in range(n_epoch):
        ptrs = np.random.permutation(data["Y"].shape[0])
        L_ep = Ns = 0
        for ptr_s in range(0, len(ptrs), batch_size):
            idx = ptrs[ptr_s : ptr_s + batch_size]
            if model_name == "diadem":
                xi = tf.cast(data["Xi"][idx], tf.float32)
                a = tf.cast(data["A"][idx, 1:], tf.float32)
                a_idx = tf.cast(data["A"][idx, 0], tf.int32) if getattr(model, "use_identity_embedding", False) else None
                y = tf.cast(data["Y"][idx], tf.float32)
                yi = tf.cast(data["Yi"][idx], tf.float32)
                ya = tf.cast(data["Ya"][idx], tf.float32)
                I_s = tf.constant(data["I"][idx].flatten().astype(np.int32))
                L_t = model.update(xi, a, yi, ya, y, I_s, update_radius, a_idx=a_idx)
            else:
                xi = tf.cast(data["Xi"][idx], tf.float32)
                a = tf.cast(data["A"][idx, :], tf.float32)
                y = tf.cast(data["Y"][idx], tf.float32)
                yi = tf.cast(data["Yi"][idx], tf.float32)
                ya = tf.cast(data["Ya"][idx], tf.float32)
                L_t = model.update(xi, a, yi, ya, y, update_radius)
            L_ep += float(L_t) * len(idx)
            Ns += len(idx)
        print(f"[{model_name}] Epoch {e}/{n_epoch - 1} L={L_ep / Ns:.4f} ({Ns} samples)")
        sys.stdout.flush()

        if e % eval_every == 0:
            print(f"[{model_name}] evaluating epoch {e} ...")
            sys.stdout.flush()
            if model_name == "diadem":
                acc, L, _, _, agg_acc, f1 = calc_stats_diadem(
                    model, data["Xi"], data["Yi"], data["Ya"], data["Y"], data["A"], data["I"],
                    batch_size,
                    all_annotator_metadata=data["annotator_metadata"],
                    all_annotator_ids=data.get("annotator_id_list"),
                )
                alpha_dict = model.get_alpha_dict()
                row = {"epoch": e, "acc": acc, "agg_acc": agg_acc, "loss": L, "f1_macro": f1, "alpha": alpha_dict}
                print(f"  Train Acc={acc:.4f} Agg={agg_acc:.4f} F1={f1:.4f}")
                print(f"  Alpha: {alpha_dict}")

                if data["dev_Y"] is not None:
                    print(f"  computing dev metrics ...")
                    sys.stdout.flush()
                    dev_acc, dev_L, _, _, dev_agg, dev_f1 = calc_stats_diadem(
                        model, data["dev_Xi"], data["dev_Yi"], data["dev_Ya"], data["dev_Y"],
                        data["dev_A"], data["dev_I"], batch_size,
                        all_annotator_metadata=data["dev_annotator_metadata"],
                        all_annotator_ids=data.get("dev_annotator_id_list"),
                    )
                    row["dev_acc"] = dev_acc
                    row["dev_agg_acc"] = dev_agg
                    row["dev_f1_macro"] = dev_f1
                    print(f"  Dev Acc={dev_acc:.4f} Agg={dev_agg:.4f} F1={dev_f1:.4f}")
            else:
                acc, L, f1 = calc_stats_disco(model, data, batch_size)
                row = {"epoch": e, "acc": acc, "loss": L, "f1_macro": f1}
                print(f"  Train Acc={acc:.4f} F1={f1:.4f}")
                if data["dev_Y"] is not None:
                    print(f"  computing dev metrics ...")
                    sys.stdout.flush()
                    dev_data = {
                        "Xi": data["dev_Xi"], "Yi": data["dev_Yi"], "Ya": data["dev_Ya"],
                        "Y": data["dev_Y"], "A": data["dev_A"], "I": data["dev_I"],
                    }
                    dev_acc, dev_L, dev_f1 = calc_stats_disco(model, dev_data, batch_size)
                    row["dev_acc"] = dev_acc
                    row["dev_f1_macro"] = dev_f1
                    print(f"  Dev Acc={dev_acc:.4f} F1={dev_f1:.4f}")
            history.append(row)
            sys.stdout.flush()

        if e % save_every == 0:
            path = data_dir / "trained_model.disco"
            save_object(model, str(path))
            print(f"[{model_name}] checkpoint saved -> {path}")
            sys.stdout.flush()

    print(f"[train] saving final model ...")
    sys.stdout.flush()
    save_object(model, str(data_dir / "trained_model.disco"))
    with open(data_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] wrote train_history.json ({len(history)} rows)")

    if model_name == "diadem":
        alpha_dict = model.get_alpha_dict()
        with open(data_dir / "alpha_weights.json", "w") as f:
            json.dump(alpha_dict, f, indent=2)
        print(f"[train] wrote alpha_weights.json: {alpha_dict}")

    print(f"[train] Saved model -> {data_dir / 'trained_model.disco'}")
    sys.stdout.flush()
