from __future__ import annotations

import unittest

import numpy as np

from config import EnvConfig

from env.action_types import (
    FastAction,
    SlowAction,
)
from env.channel import RSUChannelModel
from env.delivery.rsu_delivery import (
    compute_rsu_delivery,
)


class RSUDeliveryCommonRandomNumberTest(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.cfg = EnvConfig()

        self.num_rsu = int(
            self.cfg.num_rsu
        )

        self.num_uav = int(
            self.cfg.num_uav
        )

        self.num_user = int(
            self.cfg.num_user
        )

        self.channel = RSUChannelModel(
            self.cfg.rsu_channel
        )

        self.slow_action = SlowAction(
            rsu_scheduling=np.ones(
                (
                    self.num_rsu,
                    self.num_user,
                ),
                dtype=np.int32,
            ),
            uav_hiring=np.zeros(
                self.num_uav,
                dtype=np.int32,
            ),
            uav_scheduling=np.zeros(
                (
                    self.num_uav,
                    self.num_user,
                ),
                dtype=np.int32,
            ),
        )

    def _make_fast_action(
        self,
        *,
        dense: bool,
    ) -> FastAction:
        rsu_chunks = np.zeros(
            (
                self.num_rsu,
                self.num_user,
            ),
            dtype=np.int32,
        )

        rsu_layers = np.zeros(
            (
                self.num_rsu,
                self.num_user,
            ),
            dtype=np.int32,
        )

        if dense:
            rsu_chunks[
                :,
                :,
            ] = 1

            rsu_layers[
                :,
                :,
            ] = 1
        else:
            rsu_chunks[
                0,
                0,
            ] = 1

            rsu_layers[
                0,
                0,
            ] = 1

        return FastAction(
            rsu_chunks=rsu_chunks,
            rsu_layers=rsu_layers,
            uav_chunks=np.zeros(
                (
                    self.num_uav,
                    self.num_user,
                ),
                dtype=np.int32,
            ),
            uav_layers=np.zeros(
                (
                    self.num_uav,
                    self.num_user,
                ),
                dtype=np.int32,
            ),
            uav_power=np.zeros(
                (
                    self.num_uav,
                    self.num_user,
                ),
                dtype=np.float32,
            ),
            uav_charge=np.zeros(
                self.num_uav,
                dtype=np.int32,
            ),
            playback=np.ones(
                self.num_user,
                dtype=np.float32,
            ),
            rsu_user_distance=np.full(
                (
                    self.num_rsu,
                    self.num_user,
                ),
                15.0,
                dtype=np.float32,
            ),
            uav_user_distance=np.full(
                (
                    self.num_uav,
                    self.num_user,
                ),
                15.0,
                dtype=np.float32,
            ),
            residual_users=np.ones(
                self.num_user,
                dtype=np.int32,
            ),
            user_virtual_queue=np.zeros(
                self.num_user,
                dtype=np.float32,
            ),
            requested_content=np.zeros(
                self.num_user,
                dtype=np.int32,
            ),
            uav_cached_content=np.zeros(
                self.num_uav,
                dtype=np.int32,
            ),
        )

    def test_channel_rng_consumption_is_policy_independent(
        self,
    ) -> None:
        sparse_rng = np.random.default_rng(
            123456
        )

        dense_rng = np.random.default_rng(
            123456
        )

        sparse_result = compute_rsu_delivery(
            cfg=self.cfg,
            slow_act=self.slow_action,
            fast_act=self._make_fast_action(
                dense=False
            ),
            rsu_channel=self.channel,
            rng=sparse_rng,
        )

        dense_result = compute_rsu_delivery(
            cfg=self.cfg,
            slow_act=self.slow_action,
            fast_act=self._make_fast_action(
                dense=True
            ),
            rsu_channel=self.channel,
            rng=dense_rng,
        )

        # 공통 link (0, 0)은 Fast request mask가 달라도
        # 동일한 channel realization을 가져야 한다.
        self.assertAlmostEqual(
            float(
                sparse_result
                .raw_channel_gain[0, 0]
            ),
            float(
                dense_result
                .raw_channel_gain[0, 0]
            ),
            places=7,
        )

        # 함수 실행 이후 RNG state 역시 동일해야 한다.
        # 즉 policy가 channel RNG 소비량을 바꾸지 못해야 한다.
        self.assertAlmostEqual(
            float(
                sparse_rng.random()
            ),
            float(
                dense_rng.random()
            ),
            places=15,
        )


if __name__ == "__main__":
    unittest.main()