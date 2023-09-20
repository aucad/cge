import numpy as np
import autograd.numpy as anp
import pandas as pd

class ProblemConstraints(object):

    _custom_constraints_script = None
    _custom_constraints_count = None
    
    def __init__(self):
        pass

    @staticmethod 
    def set_custom_constraints(count, script_path):
        ProblemConstraints._custom_constraints_count = count

        with open(script_path) as myfile:
            script = myfile.read()
            ProblemConstraints._custom_constraints_script = script

    @property
    def nb_constraints(self):
        if ProblemConstraints._custom_constraints_count is None:
            return 10
        else:
            return ProblemConstraints._custom_constraints_count

    def evaluate(self, x_ml):

        # ----- PARAMETERS

        tol = 1e-3


        # ------ CUSTOM CONSTRAINTS

        if ProblemConstraints._custom_constraints_script:
            constraints = []
            exec(ProblemConstraints._custom_constraints_script)
            return anp.column_stack(constraints) - tol


        # installment = loan_amount * int_rate (1 + int_rate) ^ term / ((1+int_rate) ^ term - 1)
        calculated_installment = (
            np.ceil(
                100
                * (x_ml[:, 0] * (x_ml[:, 2] / 1200) * (1 + x_ml[:, 2] / 1200) ** x_ml[:, 1])
                / ((1 + x_ml[:, 2] / 1200) ** x_ml[:, 1] - 1)
            )
            / 100
        )
        g41 = np.absolute(x_ml[:, 3] - calculated_installment)

        # open_acc <= total_acc
        g42 = x_ml[:, 10] - x_ml[:, 14]

        # pub_rec_bankruptcies <= pub_rec
        g43 = x_ml[:, 16] - x_ml[:, 11]

        # term = 36 or term = 60
        g44 = np.absolute((36 - x_ml[:, 1]) * (60 - x_ml[:, 1]))

        # ratio_loan_amnt_annual_inc
        g45 = np.absolute(x_ml[:, 20] - x_ml[:, 0] / x_ml[:, 6])

        # ratio_open_acc_total_acc
        g46 = np.absolute(x_ml[:, 21] - x_ml[:, 10] / x_ml[:, 14])

        # diff_issue_d_earliest_cr_line
        g47 = np.absolute(
            x_ml[:, 22]
            - (self._date_feature_to_month(x_ml[:, 7]) - self._date_feature_to_month(x_ml[:, 9]))
        )

        # ratio_pub_rec_diff_issue_d_earliest_cr_line
        g48 = np.absolute(x_ml[:, 23] - x_ml[:, 11] / x_ml[:, 22])

        # ratio_pub_rec_bankruptcies_pub_rec
        g49 = np.absolute(x_ml[:, 24] - x_ml[:, 16] / x_ml[:, 22])

        # ratio_pub_rec_bankruptcies_pub_rec
        ratio_mask = x_ml[:, 11] == 0
        ratio = np.empty(x_ml.shape[0])
        ratio = np.ma.masked_array(ratio, mask=ratio_mask, fill_value=-1).filled()
        ratio[~ratio_mask] = x_ml[~ratio_mask, 16] / x_ml[~ratio_mask, 11]
        ratio[ratio == np.inf] = -1
        ratio[np.isnan(ratio)] = -1
        g410 = np.absolute(x_ml[:, 25] - ratio)

        return anp.column_stack([g41, g42, g43, g44, g45, g46, g47, g48, g49, g410]) - tol

    def _date_feature_to_month(self,feature):
        return np.floor(feature / 100) * 12 + (feature % 100)


class ProblemEvaluation(object):
    
    def __init__(self, result, encoder, initial_state, threshold, model):
        self.result = result
        self.encoder, self.initial_state, self.threshold, self.model = encoder, initial_state, threshold, model
        
    def calculate_objectives(self, return_valid=False):

        result, encoder, initial_state, threshold, model = self.result,self.encoder, self.initial_state, self.threshold, self.model
        
        CVs = np.array(list(map(lambda x: x.CV[0], result.pop)))
        Xs = np.array(list(map(lambda x: x.X, result.pop))).astype(np.float64)
        respectsConstraints = (CVs == 0).astype(np.int64)
        Xs_ml = encoder.from_genetic_to_ml(initial_state, Xs).astype(np.float64)
        isMisclassified = np.array(model.predict_proba(Xs_ml)[:, 1] < threshold).astype(
            np.int64
        )
        isBigAmount = (Xs[:, 0] >= 10000).astype(np.int64)

        o3 = respectsConstraints * isMisclassified
        o4 = o3 * isBigAmount
        objectives = np.array([respectsConstraints, isMisclassified, o3, o4])
        objectives = objectives.sum(axis=1)
        objectives = (objectives > 0).astype(np.int64)

        if return_valid:
            return list(Xs_ml[o4]), objectives
        else:
            return objectives