"""
DisCo model with learnable demographic importance weights (alpha).

Annotator encoding modes (when use_identity_embedding=True):

  identity_alpha_mode="compete" (legacy joint softmax):
    z_a = α_id·Wa[id] + Σ_d α_d·(W_d @ a_d)
    softmax(α_id, α_1…α_D) = 1

  identity_alpha_mode="additive" (ID full-strength; alphas only on demos):
    z_a = Wa[id] + Σ_d α_d·(W_d @ a_d)
    softmax(α_1…α_D) = 1

Without identity: z_a = Σ_d α_d·(W_d @ a_d) only.
"""
import tensorflow as tf
import numpy as np
from diadem.utils.utils import (
    softmax, init_weights, calc_catNLL, D_KL, drop_out, l1_l2_norm_calculation,
)

_VALID_IDENTITY_ALPHA_MODES = ("compete", "additive")


class DISCO_DemographicWeighted:
    """
    DiADEM: identity embedding (optional) + per-demographic projections with alphas.
    """

    # ── Backward-compat defaults for attributes added after old models were pickled ──
    # __getattr__ is ONLY called when normal lookup fails (i.e. attr not in __dict__),
    # so this never interferes with models that already have these attributes set.
    _LEGACY_DEFAULTS = {
        "use_identity_embedding": False,
        "use_interaction_features": False,
        "use_disagreement_detection": False,
        "identity_alpha_mode": "compete",
        "interaction_dim": 64,
        "disagreement_lambda": 0.5,
        "n_annotators": 0,
        "metadata_feature_names": [],
        "metadata_feature_dims": [],
    }

    def __getattr__(self, name):
        if name in DISCO_DemographicWeighted._LEGACY_DEFAULTS:
            return DISCO_DemographicWeighted._LEGACY_DEFAULTS[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __init__(self, xi_dim, yi_dim, ya_dim, y_dim, a_dim, lat_i_dim=20, lat_a_dim=30,
                 lat_dim=10, act_fx="softsign", init_type="gaussian", name="var",
                 lat_fusion_type="sum", drop_p=0.0, gamma_i=1.0, gamma_a=1.0,
                 l1_norm=0.0, l2_norm=0.0,
                 metadata_feature_names=None, metadata_feature_dims=None,
                 use_interaction_features=True, interaction_dim=64,
                 use_disagreement_detection=True, disagreement_lambda=0.5,
                 use_identity_embedding=True, identity_alpha_mode="additive",
                 n_annotators=None):
        self.name = name
        self.seed = 69
        self.gamma_i = gamma_i
        self.gamma_a = gamma_a
        self.lat_fusion_type = lat_fusion_type
        self.a_dim = a_dim
        self.y_dim = y_dim
        self.xi_dim = xi_dim
        self.yi_dim = yi_dim
        self.ya_dim = ya_dim
        self.lat_dim = lat_dim
        self.lat_i_dim = lat_i_dim
        self.lat_a_dim = lat_a_dim
        self.drop_p = drop_p
        self.l1_norm = l1_norm
        self.l2_norm = l2_norm
        self.metadata_feature_names = metadata_feature_names or []
        self.metadata_feature_dims = metadata_feature_dims or []
        self.use_interaction_features = use_interaction_features
        self.interaction_dim = interaction_dim
        self.use_disagreement_detection = use_disagreement_detection
        self.disagreement_lambda = disagreement_lambda
        self.use_identity_embedding = bool(use_identity_embedding)
        mode = str(identity_alpha_mode or "additive").lower().strip()
        if mode not in _VALID_IDENTITY_ALPHA_MODES:
            raise ValueError(
                f"identity_alpha_mode must be one of {_VALID_IDENTITY_ALPHA_MODES}, got {identity_alpha_mode!r}"
            )
        # Only meaningful when identity is on; demos-only otherwise.
        self.identity_alpha_mode = mode if self.use_identity_embedding else "compete"
        self.n_annotators = int(n_annotators) if n_annotators is not None else 0

        self.effective_meta_dim = sum(self.metadata_feature_dims) if self.metadata_feature_dims else a_dim
        if self.effective_meta_dim == 0:
            self.effective_meta_dim = a_dim
        if not self.metadata_feature_names or not self.metadata_feature_dims:
            self.metadata_feature_names = ["metadata"]
            self.metadata_feature_dims = [a_dim]

        if self.lat_fusion_type != "concat":
            self.lat_a_dim = self.lat_i_dim

        self.act_fx = act_fx
        if act_fx == "tanh":
            self.fx = tf.nn.tanh
        elif act_fx == "sigmoid":
            self.fx = tf.nn.sigmoid
        elif act_fx == "relu":
            self.fx = tf.nn.relu
        elif act_fx == "softsign":
            self.fx = tf.nn.softsign
        else:
            self.fx = tf.identity
        self.fx_y = softmax
        self.fx_yi = softmax
        self.fx_ya = softmax

        stddev = 0.05
        self.theta_y = []

        self.Wi = init_weights(init_type, [self.xi_dim, self.lat_i_dim], self.seed, stddev=stddev)
        self.Wi = tf.Variable(self.Wi, name="Wi", dtype=tf.float32)
        self.theta_y.append(self.Wi)

        self.Wa_identity = None
        if self.use_identity_embedding:
            if self.n_annotators <= 0:
                raise ValueError("use_identity_embedding=True requires n_annotators > 0")
            self.Wa_identity = init_weights(
                init_type, [self.n_annotators, self.lat_a_dim], self.seed, stddev=stddev
            )
            self.Wa_identity = tf.Variable(self.Wa_identity, name="Wa_identity", dtype=tf.float32)
            self.theta_y.append(self.Wa_identity)

        self.W_demographics = []
        self.demographic_offsets = []
        self.Wa_fallback = None
        offset = 0
        for d, (fname, dim) in enumerate(zip(self.metadata_feature_names, self.metadata_feature_dims)):
            W_d = init_weights(init_type, [dim, self.lat_a_dim], self.seed + d, stddev=stddev)
            W_d = tf.Variable(W_d, name=f"W_{fname}", dtype=tf.float32)
            self.W_demographics.append(W_d)
            self.theta_y.append(W_d)
            self.demographic_offsets.append((offset, offset + dim))
            offset += dim
        self.num_demographics = len(self.W_demographics)
        if self.num_demographics == 0:
            self.Wa_fallback = init_weights(init_type, [self.effective_meta_dim, self.lat_a_dim], self.seed, stddev=stddev)
            self.Wa_fallback = tf.Variable(self.Wa_fallback, name="Wa", dtype=tf.float32)
            self.theta_y.append(self.Wa_fallback)

        # Additive: alphas only over demos. Compete: Identity shares softmax with demos.
        id_in_alpha = self.use_identity_embedding and self.identity_alpha_mode == "compete"
        n_demo_alpha = max(self.num_demographics, 1 if self.Wa_fallback is not None else 0)
        n_alpha = n_demo_alpha + (1 if id_in_alpha else 0)
        if n_alpha == 0:
            n_alpha = 1
        self.alpha_raw = tf.Variable(tf.zeros([n_alpha]), name="alpha_raw", dtype=tf.float32)
        self.theta_y.append(self.alpha_raw)
        self.alpha_names = (
            (["Identity"] if id_in_alpha else [])
            + list(self.metadata_feature_names)
        )

        if self.use_interaction_features:
            self.W_interaction = init_weights(init_type, [self.lat_i_dim + self.lat_a_dim, self.interaction_dim],
                                             self.seed, stddev=stddev)
            self.W_interaction = tf.Variable(self.W_interaction, name="W_interaction", dtype=tf.float32)
            self.theta_y.append(self.W_interaction)
            self.W_hadamard_i = init_weights(init_type, [self.lat_i_dim, self.interaction_dim],
                                            self.seed, stddev=stddev)
            self.W_hadamard_i = tf.Variable(self.W_hadamard_i, name="W_hadamard_i", dtype=tf.float32)
            self.theta_y.append(self.W_hadamard_i)
            self.W_hadamard_a = init_weights(init_type, [self.lat_a_dim, self.interaction_dim],
                                            self.seed + 1, stddev=stddev)
            self.W_hadamard_a = tf.Variable(self.W_hadamard_a, name="W_hadamard_a", dtype=tf.float32)
            self.theta_y.append(self.W_hadamard_a)
            if self.lat_fusion_type != "concat":
                self.W_interaction_proj = init_weights(init_type, [self.interaction_dim * 2, self.lat_i_dim],
                                                      self.seed, stddev=stddev)
                self.W_interaction_proj = tf.Variable(self.W_interaction_proj, name="W_interaction_proj", dtype=tf.float32)
                self.theta_y.append(self.W_interaction_proj)

        bot_dim = self.lat_i_dim
        if self.lat_fusion_type == "concat":
            bot_dim = self.lat_i_dim + self.lat_a_dim
            if self.use_interaction_features:
                bot_dim += self.interaction_dim * 2

        self.Wp = init_weights(init_type, [bot_dim, self.lat_dim], self.seed, stddev=stddev)
        self.Wp = tf.Variable(self.Wp, name="Wp", dtype=tf.float32)
        self.theta_y.append(self.Wp)
        self.We = init_weights(init_type, [self.lat_dim, self.lat_dim], self.seed, stddev=stddev)
        self.We = tf.Variable(self.We, name="We", dtype=tf.float32)
        self.theta_y.append(self.We)
        self.Wy = init_weights(init_type, [self.lat_dim, self.y_dim], self.seed, stddev=stddev)
        self.Wy = tf.Variable(self.Wy, name="Wy", dtype=tf.float32)
        self.theta_y.append(self.Wy)
        self.Wyi = init_weights(init_type, [self.lat_dim, self.yi_dim], self.seed, stddev=stddev)
        self.Wyi = tf.Variable(self.Wyi, name="Wyi", dtype=tf.float32)
        self.theta_y.append(self.Wyi)
        self.Wya = init_weights(init_type, [self.lat_dim, self.ya_dim], self.seed, stddev=stddev)
        self.Wya = tf.Variable(self.Wya, name="Wya", dtype=tf.float32)
        self.theta_y.append(self.Wya)

        self.Wyi_a = init_weights(init_type, [self.lat_a_dim, self.yi_dim], self.seed + 2, stddev=stddev * 2)
        self.Wyi_a = tf.Variable(self.Wyi_a, name="Wyi_a", dtype=tf.float32)
        self.theta_y.append(self.Wyi_a)

        if self.use_disagreement_detection:
            self.W_disagreement = init_weights(init_type, [self.lat_dim, 1], self.seed, stddev=stddev)
            self.W_disagreement = tf.Variable(self.W_disagreement, name="W_disagreement", dtype=tf.float32)
            self.theta_y.append(self.W_disagreement)

        self.eta_v = 0.002
        self.y_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.eta_v, beta1=0.9, beta2=0.999, epsilon=1e-7)

    def set_opt(self, opt_type, eta_v, moment_v=0.9):
        self.eta_v = eta_v
        if opt_type == "adam":
            self.y_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.eta_v, beta1=0.9, beta2=0.999, epsilon=1e-7)
        elif opt_type == "rmsprop":
            self.y_opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.eta_v, decay=0.9, momentum=moment_v, epsilon=1e-6)
        else:
            self.y_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.eta_v)

    def get_alpha(self):
        return tf.nn.softmax(self.alpha_raw)

    def get_alpha_dict(self):
        alpha = self.get_alpha().numpy()
        names = self.alpha_names
        n = min(len(alpha), len(names))
        return {names[i]: float(alpha[i]) for i in range(n)}

    def encode_i(self, xi):
        xi = tf.cast(xi, tf.float32)
        return tf.matmul(xi, self.Wi)

    def encode_a(self, a, a_idx=None):
        """
        compete:  z_a = α_id·Wa[id] + Σ α_d·m_d
        additive: z_a = Wa[id] + Σ α_d·m_d   (α only over demos)
        no-id:    z_a = Σ α_d·m_d
        """
        a = tf.cast(a, tf.float32)
        alpha = tf.nn.softmax(self.alpha_raw)
        z_parts = []
        alpha_i = 0

        if self.use_identity_embedding:
            if a_idx is None:
                raise ValueError("a_idx required when use_identity_embedding=True")
            av = tf.cast(tf.reshape(a_idx, [-1]), tf.int32)
            av = tf.clip_by_value(av, 0, self.n_annotators - 1)
            e_id = tf.nn.embedding_lookup(self.Wa_identity, av)
            if self.identity_alpha_mode == "compete":
                z_parts.append(alpha[alpha_i] * e_id)
                alpha_i += 1
            else:
                # additive: identity outside alpha
                z_parts.append(e_id)

        if self.Wa_fallback is not None:
            z_parts.append(alpha[alpha_i] * tf.matmul(a, self.Wa_fallback))
        else:
            for d in range(self.num_demographics):
                start, end = self.demographic_offsets[d]
                a_d = a[:, start:end]
                e_d = tf.matmul(a_d, self.W_demographics[d])
                z_parts.append(alpha[alpha_i + d] * e_d)

        return tf.add_n(z_parts)

    def compute_interaction_features(self, z_i, z_a):
        if not self.use_interaction_features:
            return None
        z_concat = tf.concat([z_i, z_a], axis=1)
        z_interaction = self.fx(tf.matmul(z_concat, self.W_interaction))
        z_i_proj = self.fx(tf.matmul(z_i, self.W_hadamard_i))
        z_a_proj = self.fx(tf.matmul(z_a, self.W_hadamard_a))
        z_hadamard = z_i_proj * z_a_proj
        return tf.concat([z_interaction, z_hadamard], axis=1)

    def encode(self, xi, a, a_idx=None):
        z_i = self.encode_i(xi)
        z_a = self.encode_a(a, a_idx=a_idx)
        z_interaction = self.compute_interaction_features(z_i, z_a)
        if self.lat_fusion_type == "concat":
            if z_interaction is not None:
                z_combined = tf.concat([z_i, z_a, z_interaction], axis=1)
            else:
                z_combined = tf.concat([z_i, z_a], axis=1)
        else:
            if z_interaction is not None:
                z_interaction_proj = self.fx(tf.matmul(z_interaction, self.W_interaction_proj))
                z_combined = z_i + z_a + z_interaction_proj
            else:
                z_combined = z_i + z_a
        z = self.fx(z_combined)
        z = self.transform(z)
        return z, z_a

    def transform(self, z):
        z_p = self.fx(tf.matmul(z, self.Wp))
        if self.drop_p > 0.0:
            z_p, _ = drop_out(z_p, rate=self.drop_p)
        z_e = self.fx(tf.matmul(z_p, self.We) + z_p)
        if self.drop_p > 0.0:
            z_e, _ = drop_out(z_e, rate=self.drop_p)
        return z_e

    def compute_disagreement_loss(self, I_s, y, yi_prob):
        if not self.use_disagreement_detection:
            return 0.0
        I_flat = tf.reshape(tf.cast(I_s, tf.int32), [-1])
        _, seg_ids = tf.unique(I_flat)
        num_segments = tf.reduce_max(seg_ids) + 1

        actual_class = tf.cast(tf.argmax(y, axis=1), tf.float32)
        pred_class = tf.cast(tf.argmax(yi_prob, axis=1), tf.float32)

        count = tf.math.unsorted_segment_sum(tf.ones_like(actual_class), seg_ids, num_segments) + 1e-8
        mean_act = tf.math.unsorted_segment_sum(actual_class, seg_ids, num_segments) / count
        mean_act2 = tf.math.unsorted_segment_sum(actual_class ** 2, seg_ids, num_segments) / count
        var_act = mean_act2 - mean_act ** 2

        mean_pred = tf.math.unsorted_segment_sum(pred_class, seg_ids, num_segments) / count
        mean_pred2 = tf.math.unsorted_segment_sum(pred_class ** 2, seg_ids, num_segments) / count
        var_pred = mean_pred2 - mean_pred ** 2

        mask = tf.cast(count > 1.5, tf.float32)
        diff = tf.abs(var_act - var_pred) * mask
        n_valid = tf.reduce_sum(mask) + 1e-8
        loss = tf.reduce_sum(diff) / n_valid
        return self.disagreement_lambda * tf.cond(n_valid > 1.0, lambda: loss, lambda: 0.0)

    def decode_yi(self, z, z_a=None):
        y_logits = tf.matmul(z, self.Wyi)
        if z_a is not None and hasattr(self, "Wyi_a"):
            y_logits = y_logits + tf.matmul(z_a, self.Wyi_a)
        return self.fx_yi(y_logits), y_logits

    def decode_ya(self, z):
        y_logits = tf.matmul(z, self.Wya)
        return self.fx_ya(y_logits), y_logits

    def decode_y(self, z):
        y_logits = tf.matmul(z, self.Wy)
        return self.fx_y(y_logits), y_logits

    def calc_loss(self, I_s, y, yi, ya, y_prob, yi_prob, ya_prob):
        Ly = tf.reduce_mean(calc_catNLL(target=y, prob=y_prob, keep_batch=True))
        Lyi = D_KL(yi, yi_prob) * self.gamma_i
        Lya = D_KL(ya, ya_prob) * self.gamma_a
        L_disagreement = self.compute_disagreement_loss(I_s, y, yi_prob)
        l1 = l1_l2_norm_calculation(self.theta_y, 1, y.shape[0] * 1.0) * self.l1_norm if self.l1_norm > 0 else 0.0
        l2 = l1_l2_norm_calculation(self.theta_y, 2, y.shape[0] * 1.0) * self.l2_norm if self.l2_norm > 0 else 0.0
        return Ly + Lyi + Lya + l1 + l2 + L_disagreement, Ly, Lyi, Lya

    def update(self, xi, a, yi, ya, y, I_s, update_radius=-1.0, a_idx=None):
        with tf.GradientTape() as tape:
            z, z_a = self.encode(xi, a, a_idx=a_idx)
            yi_prob, _ = self.decode_yi(z, z_a)
            ya_prob, _ = self.decode_ya(z)
            y_prob, _ = self.decode_y(z)
            L_t, Ly, Lyi, Lya = self.calc_loss(I_s, y, yi, ya, y_prob, yi_prob, ya_prob)
        grads = tape.gradient(L_t, self.theta_y)
        if update_radius > 0:
            grads = [tf.clip_by_value(g, -update_radius, update_radius) for g in grads]
        self.y_opt.apply_gradients(zip(grads, self.theta_y))
        return L_t

    def decode_y_ensemble(self, xi, all_annotator_metadata=None, all_annotator_ids=None):
        prev_drop = self.drop_p
        self.drop_p = 0.0
        z_i = self.encode_i(xi)
        batch_size = int(xi.shape[0])
        if all_annotator_metadata is not None:
            num_annotators = int(all_annotator_metadata.shape[0])
            tiled_z_i = tf.tile(z_i, [num_annotators, 1])
            tiled_meta = tf.tile(tf.cast(all_annotator_metadata, tf.float32), [batch_size, 1])
            tiled_ids = None
            if self.use_identity_embedding:
                if all_annotator_ids is None:
                    tiled_ids = tf.zeros([batch_size * num_annotators], dtype=tf.int32)
                else:
                    ids = tf.cast(tf.reshape(all_annotator_ids, [-1]), tf.int32)
                    tiled_ids = tf.tile(ids, [batch_size])
            z_a = self.encode_a(tiled_meta, a_idx=tiled_ids)
            z_interaction = self.compute_interaction_features(tiled_z_i, z_a)
            if self.lat_fusion_type == "concat":
                if z_interaction is not None:
                    z_combined = tf.concat([tiled_z_i, z_a, z_interaction], axis=1)
                else:
                    z_combined = tf.concat([tiled_z_i, z_a], axis=1)
            else:
                if z_interaction is not None:
                    z_interaction_proj = self.fx(tf.matmul(z_interaction, self.W_interaction_proj))
                    z_combined = tiled_z_i + z_a + z_interaction_proj
                else:
                    z_combined = tiled_z_i + z_a
            z = self.fx(z_combined)
            z = self.transform(z)
            y_prob, _ = self.decode_y(z)
            y_prob = tf.reshape(y_prob, [batch_size, num_annotators, -1])
            y_prob = tf.reduce_mean(y_prob, axis=1)
        else:
            dummy = tf.zeros([batch_size, self.effective_meta_dim], dtype=tf.float32)
            z_a = self.encode_a(
                dummy,
                a_idx=tf.zeros([batch_size], dtype=tf.int32) if self.use_identity_embedding else None,
            )
            z_interaction = self.compute_interaction_features(z_i, z_a)
            if self.lat_fusion_type == "concat":
                if z_interaction is not None:
                    z_combined = tf.concat([z_i, z_a, z_interaction], axis=1)
                else:
                    z_combined = tf.concat([z_i, z_a], axis=1)
            else:
                z_combined = z_i + z_a
            z = self.fx(z_combined)
            z = self.transform(z)
            y_prob, _ = self.decode_y(z)
        self.drop_p = prev_drop
        return y_prob, None
