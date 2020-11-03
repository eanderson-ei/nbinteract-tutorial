[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eanderson-ei/sws-viz/master?filepath=notebooks%2Findicator-3-2-clean.ipynb)

# README

Informing Products

* [Last Reporting Period's MEL Annex](https://drive.google.com/file/d/17b-9c_b5_E3cS8Altpz51otB1RjmuI6i/view?usp=sharing) (page 58)
* [Input Data](https://drive.google.com/drive/folders/1729ESaOT9Oz876FsCyM5yFJbjU4mOTyf) (Google Drive link)

## Contents

**notebooks**/

* **MEL ANNEX.ipynb**: primary notebook for creating charts for the MEL Annnex.

**data/**

* **raw/**
  * **indicator-1-1.csv**: a copy paste of the PITS Tab 1.1 table starting in B11.
  * ...(same for other sheets named this same pattern)
  * **districts**: a crosswalk of districts, concept teams, and geographies for indicator 3.2
  * **pms**: a crosswalk of pm_id, pm_category and districts for indicator 3.2
  * **pm_data**: pm scores for each pm for indicator 3.2
  * **sws-data.xlsx**: an outdated .xlsx with all indicator data required to run the app
* **raw_Q2_2020**: a backup of raw/ made before any updates (created 11/3/2020).

**charts/**: output location for all charts as png

**charts_Q2_2020.zip**: charts provided to Patrick for Q2 report

*All other contents are used for an app interface that was not needed for the report*

## Indicator Notes

- Indicator 1.1 % of coalition participants reporting an improvement in WASH system understanding
  - Data not valid across concept teams because they use different methods, general might be the way to go with this one
  - Could break into three graphs to show change over time within concept team; see what this looks like when broken out by coalition
- Indicator 1.2 # of analyses conducted to improve understanding of WASH systems
  - Overall graph for number of studies over time
  - Look at data for FY 20 and see if we can broadly group by analysis type (or for over time) – this is a task for Laurel and Shawn
- Indicator 1.3 # of stakeholders reached with findings from systems analyses
  - Visually demarcate the concept teams for various geographies
- Indicator 2.1 % of coalitions that have an agreed set of needs and priorities
  - No visualization
- Indicator 3.1 % of SWS geographies showing an improvement in WASH network strength
  - No visualization
- Indicator 3.2 % average rating for each coalition progress markers for their vision of more sustainable services (The one we talked about in our last meeting)
  - Past: average scores of PMs for each concept team
  - Story we want to tell: something more similar to badges to visualize overall progress for each coalition over time; might group expect to see, like to see, and love to see categories
- Indicator 5.1 # of evidence products or activities by SWS partners
  - Show by type (academic, grey, presentation) for reporting period broken out by team

### Deployment

* Make sure app.py includes (after defining variable app)

  ```python
  server = app.server
  ```

* Create Procfile with conents

  ```
  web: gunicorn app:server
  ```

  Note no space after `app:`

* Install gunicorn if not already 

  ```bash
  pip install gunicorn
  ```

  Create requirements.txt

  * `pip freeze>requirements.txt`
  * You can also list the key dependencies in a text file. Should set up a pip or conda environment though.

* Use Heroku to deploy

  ```bash
  git add .
  git commit -m "<message>"
  git push origin master
  heroku create <app name>
  git push heroku master
  heroku ps:scale web=1
  ```

* Data are loaded directly from github repo for this version.

https://dash.plotly.com/deployment

https://towardsdatascience.com/how-to-build-a-complex-reporting-dashboard-using-dash-and-plotl-4f4257c18a7f