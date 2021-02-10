#ifndef RTS_2_DETECTPARAMS_H
#define RTS_2_DETECTPARAMS_H

class DetectParams {
 public:
  float thresh_multiplier = 5.0; /*!< multiple of MAD for detect detect */
  float dedupe_ms = 0.25; /*!< time partition around potential peaks */
  float dedupe_um = 50.0; /*!< space partition around potential peaks */
};

#endif //RTS_2_DETECTPARAMS_H
