import kfp
from kfp import dsl
from kfp.components import func_to_container_op
# import matplotlib.pyplot as plt


@func_to_container_op
def show_results(score_rf_reg: float, score_hist_gradient_boosting: float, score_decision_tree: float
, score_elastic_net: float, score_lasso: float, 
    score_ridge: float, score_polynomial : float
) -> None:
    # Given the outputs from decision_tree and logistic regression components
    # the results are shown.
    print(f"Random Forest regression (accuracy): {score_rf_reg}")
    print(f"Hist Gradient Boosting regression (accuracy): {score_hist_gradient_boosting}")
    print(f"Decision Tree regression (accuracy): {score_decision_tree}")
    print(f"Elastic Net regression (accuracy): {score_elastic_net}")
    print(f"Lasso regression (accuracy): {score_lasso}")
    print(f"Ridge regression (accuracy): {score_ridge}")
    print(f"Polynomial regression (accuracy): {score_polynomial}")
    # Best Model
    best_model = max(score_rf_reg, score_hist_gradient_boosting, score_decision_tree, score_elastic_net)
    # Switch Case for best model
    if best_model == score_rf_reg:
        print("Best Model is Random Forest" + str(best_model))
    elif best_model == score_hist_gradient_boosting:
        print("Best Model is Hist Gradient Boosting" + str(best_model))
    elif best_model == score_decision_tree:
        print("Best Model is Decision Tree" + str(best_model))
    elif best_model == score_elastic_net:
        print("Best Model is Elastic Net" + str(best_model))
    elif best_model == score_lasso:
        print("Best Model is Lasso" + str(best_model))
    elif best_model == score_polynomial:
        print("Best Model is Polynomial" + str(best_model))
    elif best_model == score_ridge:
        print("Best Model is Ridge" + str(best_model))
    else:
        print("No Best Model")

    # Print Comparative Graph
    # plt.bar(["Linear Regression", "Random Forest", 
    #         "Hist Gradient Boosting", 
    #         "Decision Tree", "Elastic Net", "Lasso", 
    #         "Polynomial", "Ridge"],
    #         [score_lin_reg, score_rf_reg, score_hist_gradient_boosting, score_decision_tree, 
    #         score_elastic_net, score_lasso, score_polynomial, score_ridge])
    # plt.title("Comparative Graph")
    # plt.xlabel("Models")
    # plt.show()



@dsl.pipeline(
    name="Pipeline",
    description="Applies Preprocess, Linear and Random Forest Regression problem.",
)
def test_pipeline():
    # # Loads the yaml manifest for each component
    # preprocess_clean = kfp.components.load_component_from_file(
    #     "preprocess_clean/preprocess_clean.yaml"
    # )
    # preprocess_split = kfp.components.load_component_from_file(
    #     "preprocess_split/preprocess_split.yaml"
    # )

    # # Loads the yaml manifest for each component
    preprocess = kfp.components.load_component_from_file(
        "preprocess/preprocess.yaml"
    )
    rf_regression = kfp.components.load_component_from_file(
        "rf_regression/rf_regression.yaml"
    )
    hist_gradient_boosting = kfp.components.load_component_from_file(
        "hist_gradient_boosting_regression/hist_gradient_boosting_regression.yaml"
    )
    decision_tree = kfp.components.load_component_from_file(
        "decision_tree_regression/decision_tree_regression.yaml"
    )
    elastic_net = kfp.components.load_component_from_file(
        "elastic_net_regression/elastic_net_regression.yaml"
    )
    lasso = kfp.components.load_component_from_file(
        "lasso_regression/lasso_regression.yaml"
    )
    ridge = kfp.components.load_component_from_file(
        "ridge_regression/ridge_regression.yaml"
    )
    polynomial = kfp.components.load_component_from_file(
        "polynomial_regression/polynomial_regression.yaml"
    )

    # Preprocess data 
    preprocess_task = preprocess()
    rf_regression_task = rf_regression(preprocess_task.output)
    hist_gradient_boosting_task = hist_gradient_boosting(preprocess_task.output)
    decision_tree = decision_tree(preprocess_task.output)
    elastic_net = elastic_net(preprocess_task.output)
    lasso = lasso(preprocess_task.output)
    ridge = ridge(preprocess_task.output)
    polynomial = polynomial(preprocess_task.output)

    # the component "show_results" is called to print the results.
    show_results(rf_regression_task.output, 
                hist_gradient_boosting_task.output, decision_tree.output, 
                elastic_net.output, lasso.output, ridge.output, polynomial.output)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(test_pipeline, "pipeline.yaml")
