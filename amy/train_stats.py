from time import strftime

from prometheus_client import Gauge

from amy.network import MLH_WEIGHT, WDL_WEIGHT

loss_gauge = Gauge("training_loss", "Training loss")
moves_accuracy_gauge = Gauge("training_move_accuracy", "Move accuracy")
moves_top5_accuracy_gauge = Gauge("training_move_top5_accuracy", "Top 5 move accuracy")
score_mae_gauge = Gauge("training_score_mae", "Score mean absolute error")
wdl_accuracy_gauge = Gauge("wdl_accuracy", "WDL accuracy")


class Stats(object):
    def __init__(self):

        self.sum_moves_accuracy = 0
        self.sum_moves_top5_accuracy = 0
        self.sum_score_mae = 0
        self.sum_loss = 0
        self.sum_wdl_accuracy = 0
        self.sum_mlh = 0
        self.sum_cnt = 0

    def __call__(self, step_output, cnt):

        loss = step_output[0]
        moves_loss = step_output[1]
        score_loss = step_output[2]
        wdl_loss = step_output[3]
        mlh_loss = step_output[4]
        reg_loss = abs(
            loss
            - moves_loss
            - score_loss
            - WDL_WEIGHT * wdl_loss
            - MLH_WEIGHT * mlh_loss
        )

        moves_accuracy = step_output[5]
        moves_top5_accuracy = step_output[6]
        score_mae = step_output[7]
        wdl_accuracy = step_output[8]
        mlh_mae = step_output[9]

        loss_gauge.set(loss)
        moves_accuracy_gauge.set(moves_accuracy * 100)
        moves_top5_accuracy_gauge.set(moves_top5_accuracy * 100)
        score_mae_gauge.set(score_mae)
        wdl_accuracy_gauge.set(wdl_accuracy * 100)

        self.sum_moves_accuracy += moves_accuracy * cnt
        self.sum_moves_top5_accuracy += moves_top5_accuracy * cnt
        self.sum_score_mae += score_mae * cnt
        self.sum_loss += loss * cnt
        self.sum_wdl_accuracy += wdl_accuracy * cnt
        self.sum_mlh += mlh_mae * cnt
        self.sum_cnt += cnt

        return (
            f"loss: {loss:.2f} = {moves_loss:.2f} + {score_loss:.3f} + {reg_loss:.3f}, "
            f"moves: {moves_accuracy * 100:4.1f}% top 5: {moves_top5_accuracy * 100:4.1f}%, "
            f"score: {score_mae:.2f}, "
            f"wdl: {wdl_accuracy * 100:4.1f}%, mlh: {mlh_mae:2.1f} || "
            f"avg: {self.sum_loss / self.sum_cnt:.3f}, {self.sum_moves_accuracy * 100 / self.sum_cnt:.2f}% "
            f"top 5: {self.sum_moves_top5_accuracy * 100 / self.sum_cnt:.2f}%, "
            f"{self.sum_score_mae / self.sum_cnt:.3f}, "
            f"wdl: {self.sum_wdl_accuracy * 100 / self.sum_cnt:.2f}% "
            f"mlh: {self.sum_mlh / self.sum_cnt:.1f}"
        )

    def write_to_file(self, model_name, filename="stats.txt"):

        with open(filename, "a") as statsfile:
            print(
                f"{strftime('%Y-%m-%d %H:%M')} [{model_name}] "
                f"{self.sum_cnt} positions: {self.sum_loss / self.sum_cnt:.3f}, "
                f"{self.sum_moves_accuracy * 100 / self.sum_cnt:.2f}% "
                f"top 5: {self.sum_moves_top5_accuracy * 100 / self.sum_cnt:.2f}%,"
                f"{self.sum_score_mae / self.sum_cnt:.3f}, "
                f"wdl: {self.sum_wdl_accuracy * 100 / self.sum_cnt:.2f}%",
                file=statsfile,
            )
