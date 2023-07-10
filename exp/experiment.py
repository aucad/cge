from collections import namedtuple
from sklearn.model_selection import KFold

from exp import Utility as Util, \
    Result, Validation, AttackRunner, ModelTraining


# TODO: apply experiment with validation (if enabled)

class Experiment:

    def __init__(self, conf):
        c_keys = ",".join(list(conf.keys()))
        self.X = None
        self.y = None
        self.cls = None
        self.folds = None
        self.attack = None
        self.attrs = []
        self.attr_ranges = {}
        self.config = (namedtuple('exp', c_keys)(**conf))
        self.stats = Result()
        self.validation = None

    @property
    def n_records(self) -> int:
        return len(self.X)

    def custom_config(self, key):
        return getattr(self.config, key) \
            if key and hasattr(self.config, key) else None

    def load_csv(self, ds_path: str, n_splits: int):
        self.attrs, rows = Util.read_dataset(ds_path)
        self.X = rows[:, :-1]
        self.y = rows[:, -1].astype(int).flatten()
        n_feat = len(self.attrs) - 1
        self.folds = [x for x in KFold(
            n_splits=n_splits, shuffle=True).split(self.X)]
        for col_i in range(n_feat):
            self.attr_ranges[col_i] = max(self.X[:, col_i])

    def run(self):
        config = self.config
        self.load_csv(config.dataset, config.folds)
        self.cls = ModelTraining(*(
            config.out, self.attrs, self.y,
            self.attr_ranges, self.custom_config('xgb')))
        self.attack = AttackRunner(*(
            config.iter, self.custom_config('zoo')))
        self.validation = Validation(
            original=self.X,
            immutable=self.custom_config('immutable'),
            constraints=self.custom_config('constraints'),
            apply=self.custom_config('apply_validation'))
        self.log_experiment_setup()
        for i, fold in enumerate(self.folds):
            self.exec_fold(i + 1, fold)
        self.log_experiment_result()
        Util.write_result(self.config.out, self.to_dict())

    def exec_fold(self, fi, f_idx):
        self.cls.reset() \
            .load(self.X.copy(), self.y.copy(), *f_idx, fi) \
            .train()
        self.stats.append_cls(self.cls)
        self.log_training_result(fi)
        self.validation.reset(self.cls.test_x.copy())
        self.attack.reset().set_cls(self.cls).run().eval()
        valid_score = self.validation.score_valid(self.attack.adv_x)
        self.stats.append_attack(self.attack, valid_score)
        self.log_fold_attack(valid_score)

    def log_experiment_setup(self):
        Util.log('Dataset', self.config.dataset)
        Util.log('Record count', len(self.X))
        Util.log('Attributes', len(self.attrs))
        Util.log('K-folds', self.config.folds)
        Util.log('Classifier', self.cls.name)
        Util.log('Classes', ", ".join(self.cls.class_names))
        Util.log('Immutable', len(self.validation.immutable))
        Util.log('Constraints', len(self.validation.constraints.keys()))
        Util.log('Validating?', not self.validation.disabled)
        Util.log('Attack', self.attack.name)
        Util.log('Attack max iter', self.attack.max_iter)

    def log_training_result(self, fold_n: int):
        print('=' * 52)
        Util.log('Fold', fold_n)
        Util.log('Accuracy', f'{self.cls.accuracy * 100:.2f} %')
        Util.log('Precision', f'{self.cls.precision * 100:.2f} %')
        Util.log('Recall', f'{self.cls.recall * 100:.2f} %')
        Util.log('F-score', f'{self.cls.f_score * 100:.2f} %')

    def log_fold_attack(self, valid_score):
        Util.logr('Evasions', self.attack.n_evasions, self.attack.n_records)
        Util.log('Valid', f'{valid_score * 100 :.2f} %')

    def log_experiment_result(self):
        print('=' * 52, '', '\nAVERAGE')
        Util.log('Accuracy', f'{self.stats.accuracy.avg :.2f} %')
        Util.log('Precision', f'{self.stats.precision.avg :.2f} %')
        Util.log('Recall', f'{self.stats.recall.avg :.2f} %')
        Util.log('F-score', f'{self.stats.f_score.avg :.2f} %')
        Util.logr('Evasions', self.stats.n_evasions.avg / 100,
                  self.stats.n_records.avg / 100)
        Util.log('Valid', f'{self.stats.valid.avg :.2f} %')

    def to_dict(self) -> dict:
        return {
            'dataset': self.config.dataset,
            'n_records': len(self.X),
            'n_attributes': len(self.attrs),
            'attrs': self.attrs,
            'attr_immutable': self.validation.immutable,
            'attr_constrained': list(self.validation.constraints.keys()),
            'classes': self.cls.class_names,
            'k_folds': self.config.folds,
            'classifier': self.config.cls,
            'attack': self.config.attack,
            'attr_ranges': dict(
                zip(self.attrs, self.attr_ranges.values())),
            'max_iter': self.attack.max_iter,
            **self.stats.to_dict()
        }
