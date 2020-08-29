## Datasets
Here's two datasets for training and testing: Covid and Woz. <br>Source of Covid: https://github.com/tanyuqian/covid-dst-data. <br>Source of Woz [Wizard-of-Oz restaurant]: Comes with GLAD original code. 
### Covid
Raw Structure: [train / dev / test] <br>
* List
   * dialogue_index
   * url
   * description
   * turns
      * List
          * doctor_text
          * system_acts
          * patient_text
          * turn_label
  
### WoZ
Raw Structure: [train / dev / test] <br>
* List
    * dialogue_idx
    * dialogue
        * turn_label
        * asr
        * system_transcript
        * turn_idx
        * belief_state
            * slots
            * acts
        * transcript
        * system_acts
