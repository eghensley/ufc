fighters = ['DROP TABLE IF EXISTS ufc.fighters;',
           
           'CREATE TABLE ufc.fighters \
            (fighter_id varchar COLLATE pg_catalog."default" NOT NULL, \
            name varchar COLLATE pg_catalog."default" NOT NULL, \
            height int, \
            reach int, \
            stance varchar COLLATE pg_catalog."default" NOT NULL, \
            dob timestamp NOT NULL, \
            CONSTRAINT fighter_pkey PRIMARY KEY (fighter_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.fighter_id_idx;',
            'CREATE INDEX fighter_id_idx \
            ON ufc.fighters USING btree \
            (fighter_id COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;',

            'DROP INDEX IF EXISTS ufc.fighter_name_idx;',
            'CREATE INDEX fighter_name_idx \
            ON ufc.fighters USING btree \
            (name COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;',
            ]


weights = ['DROP TABLE IF EXISTS ufc.weights;',
           
           'CREATE TABLE ufc.weights \
            (weight_id int, \
            weight_desc varchar COLLATE pg_catalog."default" NOT NULL, \
            CONSTRAINT weight_pk PRIMARY KEY (weight_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.weight_id_idx;',
            'CREATE INDEX weight_id_idx \
            ON ufc.weights USING btree \
            (weight_id) \
            TABLESPACE pg_default;'\
            
            'DROP INDEX IF EXISTS ufc.weight_name_idx;',
            'CREATE INDEX weight_name_idx \
            ON ufc.weights USING btree \
            (weight_desc COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;',
            ]


methods = ['DROP TABLE IF EXISTS ufc.methods;',
           
           'CREATE TABLE ufc.methods \
            (method_id int, \
            method_desc varchar COLLATE pg_catalog."default" NOT NULL, \
            CONSTRAINT method_pk PRIMARY KEY (method_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.method_id_idx;',
            'CREATE INDEX method_id_idx \
            ON ufc.methods USING btree \
            (method_id) \
            TABLESPACE pg_default;'\
            
            'DROP INDEX IF EXISTS ufc.method_name_idx;',
            'CREATE INDEX method_name_idx \
            ON ufc.methods USING btree \
            (method_desc COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;',
            ]

bout_stats = ['DROP TABLE IF EXISTS ufc.bout_stats;',
           
           'CREATE TABLE ufc.bout_stats \
            (bout_id varchar COLLATE pg_catalog."default" NOT NULL, \
            fighter_id varchar COLLATE pg_catalog."default" NOT NULL, \
            kd int NOT NULL, \
            ssa int NOT NULL, \
            sss int NOT NULL, \
            tsa int NOT NULL, \
            tss int NOT NULL, \
            sub int NOT NULL, \
            pas int NOT NULL, \
            rev int NOT NULL, \
            headssa int NOT NULL, \
            headsss int NOT NULL, \
            bodyssa int NOT NULL, \
            bodysss int NOT NULL, \
            legssa int NOT NULL, \
            legsss int NOT NULL, \
            disssa int NOT NULL, \
            dissss int NOT NULL, \
            clinssa int NOT NULL, \
            clinsss int NOT NULL, \
            gndssa int NOT NULL, \
            gndsss int NOT NULL, \
            tda int NOT NULL, \
            tds int NOT NULL, \
            CONSTRAINT bout_stat_fighter_fkey FOREIGN KEY (fighter_id) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT bout_stat_uq UNIQUE (bout_id, fighter_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.bout_id_idx;',
            'CREATE INDEX bout_id_idx \
            ON ufc.bout_stats USING btree \
            (bout_id COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;'
            ]


bout_fighter_xref = ['DROP TABLE IF EXISTS ufc.bout_fighter_xref;',
           
           'CREATE TABLE ufc.bout_fighter_xref \
            (bout_id varchar COLLATE pg_catalog."default" NOT NULL, \
            fighter_id varchar COLLATE pg_catalog."default" NOT NULL, \
            opponent_id varchar COLLATE pg_catalog."default" NOT NULL, \
            CONSTRAINT bout_fighter_xref_fighter_fkey FOREIGN KEY (fighter_id) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT bout_fighter_xref_opponent_fkey FOREIGN KEY (opponent_id) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT bout_fighter_xref_bout_fkey FOREIGN KEY (bout_id) \
            REFERENCES ufc.bouts (bout_id), \
            CONSTRAINT bout_fighter_xref_opponent_uq UNIQUE (bout_id, opponent_id), \
            CONSTRAINT bout_fighter_xref_fighter_uq UNIQUE (bout_id, fighter_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            ]


bout_corners = ['DROP TABLE IF EXISTS ufc.bout_corners;',
           
           'CREATE TABLE ufc.bout_corners \
            (bout_id varchar COLLATE pg_catalog."default" NOT NULL, \
            red_corner varchar COLLATE pg_catalog."default" NOT NULL, \
            blue_corner varchar COLLATE pg_catalog."default" NOT NULL, \
            CONSTRAINT bout_corners_red_fkey FOREIGN KEY (red_corner) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT bout_corners_blue_fkey FOREIGN KEY (blue_corner) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT bout_corners_bout_fkey FOREIGN KEY (bout_id) \
            REFERENCES ufc.bouts (bout_id), \
            CONSTRAINT bout_corners_uq UNIQUE (bout_id, red_corner, blue_corner)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            ]


bout_predictions = ['DROP TABLE IF EXISTS ufc.bout_predictions;',
           
           'CREATE TABLE ufc.bout_predictions \
            (bout_id varchar COLLATE pg_catalog."default" NOT NULL, \
            winner varchar COLLATE pg_catalog."default" NOT NULL, \
            loser varchar COLLATE pg_catalog."default" NOT NULL, \
            winner_score float NOT NULL, \
            length int NOT NULL, \
            CONSTRAINT bout_pred_winner_fkey FOREIGN KEY (winner) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT bout_pred_loser_fkey FOREIGN KEY (loser) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT bout_pred_bout_fkey FOREIGN KEY (bout_id) \
            REFERENCES ufc.bouts (bout_id), \
            CONSTRAINT bout_pred_uq UNIQUE (bout_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.bout_prediction_id_idx;',
            'CREATE INDEX bout_prediction_id_idx \
            ON ufc.bout_predictions USING btree \
            (bout_id COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.bout_prediction_winner_idx;',
            'CREATE INDEX bout_prediction_winner_idx \
            ON ufc.bout_predictions USING btree \
            (winner COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;'

            'DROP INDEX IF EXISTS ufc.bout_prediction_loser_idx;',
            'CREATE INDEX bout_prediction_loser_idx \
            ON ufc.bout_predictions USING btree \
            (loser COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;'
            ]


bout_results = ['DROP TABLE IF EXISTS ufc.bout_results;',
           
           'CREATE TABLE ufc.bout_results \
            (bout_id varchar COLLATE pg_catalog."default" NOT NULL, \
            winner varchar COLLATE pg_catalog."default" NOT NULL, \
            loser varchar COLLATE pg_catalog."default" NOT NULL, \
            method_id int NOT NULL, \
            rounds int NOT NULL, \
            time int NOT NULL, \
            length int NOT NULL, \
            CONSTRAINT bout_res_winner_fkey FOREIGN KEY (winner) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT bout_res_loser_fkey FOREIGN KEY (loser) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT bout_res_meth_fkey FOREIGN KEY (method_id) \
            REFERENCES ufc.methods (method_id), \
            CONSTRAINT bout_res_bout_fkey FOREIGN KEY (bout_id) \
            REFERENCES ufc.bouts (bout_id), \
            CONSTRAINT bout_res_uq UNIQUE (bout_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.bout_res_id_idx;',
            'CREATE INDEX bout_res_id_idx \
            ON ufc.bout_results USING btree \
            (bout_id COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.bout_res_winner_idx;',
            'CREATE INDEX bout_res_winner_idx \
            ON ufc.bout_results USING btree \
            (winner COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;'

            'DROP INDEX IF EXISTS ufc.bout_res_loser_idx;',
            'CREATE INDEX bout_res_loser_idx \
            ON ufc.bout_results USING btree \
            (loser COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;'
            ]


bouts = ['DROP TABLE IF EXISTS ufc.bouts;',
           
           'CREATE TABLE ufc.bouts \
            (bout_id varchar COLLATE pg_catalog."default" NOT NULL, \
            fight_id varchar COLLATE pg_catalog."default" NOT NULL, \
            weight_id int NOT NULL, \
            champ bool NOT NULL, \
            CONSTRAINT bout_fight_fkey FOREIGN KEY (fight_id) \
            REFERENCES ufc.fights (fight_id), \
            CONSTRAINT bout_weight_fkey FOREIGN KEY (weight_id) \
            REFERENCES ufc.weights (weight_id), \
            CONSTRAINT bout_pk PRIMARY KEY (bout_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.bout_id_idx;',
            'CREATE INDEX bout_id_idx \
            ON ufc.bouts USING btree \
            (bout_id COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;'
            ]


fights = ['DROP TABLE IF EXISTS ufc.fights;',
           
           'CREATE TABLE ufc.fights \
            (fight_id varchar COLLATE pg_catalog."default" NOT NULL, \
            name varchar COLLATE pg_catalog."default" NOT NULL, \
            country_id int, \
            state_id int, \
            city_id int, \
            date timestamp NOT NULL, \
            CONSTRAINT fight_country_fkey FOREIGN KEY (country_id) \
            REFERENCES ufc.countries (country_id), \
            CONSTRAINT fight_state_fkey FOREIGN KEY (state_id) \
            REFERENCES ufc.states (state_id), \
            CONSTRAINT fight_city_fkey FOREIGN KEY (city_id) \
            REFERENCES ufc.cities (city_id), \
            CONSTRAINT fight_pkey PRIMARY KEY (fight_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.fight_id_idx;',
            'CREATE INDEX fight_id_idx \
            ON ufc.fights USING btree \
            (fight_id COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;',

            'DROP INDEX IF EXISTS ufc.fight_name_idx;',
            'CREATE INDEX fight_name_idx \
            ON ufc.fights USING btree \
            (name COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;',
            ]


countries = ['DROP TABLE IF EXISTS ufc.countries;',
           
           'CREATE TABLE ufc.countries \
            (country_id int NOT NULL, \
            country_name varchar COLLATE pg_catalog."default" NOT NULL, \
            CONSTRAINT countries_pkey PRIMARY KEY (country_id), \
            CONSTRAINT countries_uq UNIQUE (country_name)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.country_id_idx;',
            'CREATE INDEX country_id_idx \
            ON ufc.countries USING btree \
            (country_id) \
            TABLESPACE pg_default;',

            'DROP INDEX IF EXISTS ufc.country_name_idx;',
            'CREATE INDEX country_name_idx \
            ON ufc.countries USING btree \
            (country_name COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;',
            ]

states = ['DROP TABLE IF EXISTS ufc.states;',
           
           'CREATE TABLE ufc.states \
            (state_id int, \
            state_name varchar COLLATE pg_catalog."default" NOT NULL, \
            country_id int NOT NULL, \
            CONSTRAINT state_country_fkey FOREIGN KEY (country_id) \
            REFERENCES ufc.countries (country_id), \
            CONSTRAINT states_pkey PRIMARY KEY (state_id), \
            CONSTRAINT states_uq UNIQUE (state_name)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.states_id_idx;',
            'CREATE INDEX states_id_idx \
            ON ufc.states USING btree \
            (state_id) \
            TABLESPACE pg_default;',

            'DROP INDEX IF EXISTS ufc.states_name_idx;',
            'CREATE INDEX states_name_idx \
            ON ufc.states USING btree \
            (state_name COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;',
            ]

cities = ['DROP TABLE IF EXISTS ufc.cities;',
           
           'CREATE TABLE ufc.cities \
            (city_id int, \
            city_name varchar COLLATE pg_catalog."default" NOT NULL, \
            country_id int NOT NULL, \
            state_id int, \
            CONSTRAINT city_country_fkey FOREIGN KEY (country_id) \
            REFERENCES ufc.countries (country_id), \
            CONSTRAINT city_states_fkey FOREIGN KEY (state_id) \
            REFERENCES ufc.states (state_id), \
            CONSTRAINT city_pkey PRIMARY KEY (city_id), \
            CONSTRAINT city_uq UNIQUE (city_name)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            
            'DROP INDEX IF EXISTS ufc.cities_id_idx;',
            'CREATE INDEX cities_id_idx \
            ON ufc.cities USING btree \
            (city_id) \
            TABLESPACE pg_default;',

            'DROP INDEX IF EXISTS ufc.city_name_idx;',
            'CREATE INDEX city_name_idx \
            ON ufc.cities USING btree \
            (city_name COLLATE pg_catalog."default" text_pattern_ops) \
            TABLESPACE pg_default;',
            ]


avg_stats = ['DROP TABLE IF EXISTS ufc.avg_stats;',
           
           'CREATE TABLE ufc.avg_stats \
            (fighter_id varchar COLLATE pg_catalog."default" NOT NULL, \
             bout_id varchar COLLATE pg_catalog."default" NOT NULL, \
              avg_o_kd float NOT NULL, \
              avg_o_ssa float NOT NULL, \
              avg_o_sss float NOT NULL, \
              avg_o_tsa float NOT NULL, \
              avg_o_tss float NOT NULL, \
              avg_o_sub float NOT NULL, \
              avg_o_pas float NOT NULL, \
              avg_o_rev float NOT NULL, \
              avg_o_headssa float NOT NULL, \
              avg_o_headsss float NOT NULL, \
              avg_o_bodyssa float NOT NULL, \
              avg_o_bodysss float NOT NULL, \
              avg_o_legssa float NOT NULL, \
              avg_o_legsss float NOT NULL, \
              avg_o_disssa float NOT NULL, \
              avg_o_dissss float NOT NULL, \
              avg_o_clinssa float NOT NULL, \
              avg_o_clinsss float NOT NULL, \
              avg_o_gndssa float NOT NULL, \
              avg_o_gndsss float NOT NULL, \
              avg_o_tda float NOT NULL, \
              avg_o_tds float NOT NULL, \
              avg_d_kd float NOT NULL, \
              avg_d_ssa float NOT NULL, \
              avg_d_sss float NOT NULL, \
              avg_d_tsa float NOT NULL, \
              avg_d_tss float NOT NULL, \
              avg_d_sub float NOT NULL, \
              avg_d_pas float NOT NULL, \
              avg_d_rev float NOT NULL, \
              avg_d_headssa float NOT NULL, \
              avg_d_headsss float NOT NULL, \
              avg_d_bodyssa float NOT NULL, \
              avg_d_bodysss float NOT NULL, \
              avg_d_legssa float NOT NULL, \
              avg_d_legsss float NOT NULL, \
              avg_d_disssa float NOT NULL, \
              avg_d_dissss float NOT NULL, \
              avg_d_clinssa float NOT NULL, \
              avg_d_clinsss float NOT NULL, \
              avg_d_gndssa float NOT NULL, \
              avg_d_gndsss float NOT NULL, \
              avg_d_tda float NOT NULL, \
              avg_d_tds float NOT NULL, \
            CONSTRAINT avg_stats_fighter_fkey FOREIGN KEY (fighter_id) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT avg_stats_bout_fkey FOREIGN KEY (bout_id) \
            REFERENCES ufc.bouts (bout_id), \
            CONSTRAINT avg_stats_uq UNIQUE (fighter_id, bout_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            ]
 

adj_stats = ['DROP TABLE IF EXISTS ufc.adj_stats;',
           
           'CREATE TABLE ufc.adj_stats \
            (fighter_id varchar COLLATE pg_catalog."default" NOT NULL, \
             bout_id varchar COLLATE pg_catalog."default" NOT NULL, \
            adj_d_bodyssa float NOT NULL, \
            adj_d_bodysss float NOT NULL, \
            adj_d_clinssa float NOT NULL, \
            adj_d_clinsss float NOT NULL, \
            adj_d_disssa float NOT NULL, \
            adj_d_dissss float NOT NULL, \
            adj_d_gndssa float NOT NULL, \
            adj_d_gndsss float NOT NULL, \
            adj_d_headssa float NOT NULL, \
            adj_d_headsss float NOT NULL, \
            adj_d_kd float NOT NULL, \
            adj_d_legssa float NOT NULL, \
            adj_d_legsss float NOT NULL, \
            adj_d_pas float NOT NULL, \
            adj_d_rev float NOT NULL, \
            adj_d_ssa float NOT NULL, \
            adj_d_sss float NOT NULL, \
            adj_d_sub float NOT NULL, \
            adj_d_tda float NOT NULL, \
            adj_d_tds float NOT NULL, \
            adj_d_tsa float NOT NULL, \
            adj_d_tss float NOT NULL, \
            adj_o_bodyssa float NOT NULL, \
            adj_o_bodysss float NOT NULL, \
            adj_o_clinssa float NOT NULL, \
            adj_o_clinsss float NOT NULL, \
            adj_o_disssa float NOT NULL, \
            adj_o_dissss float NOT NULL, \
            adj_o_gndssa float NOT NULL, \
            adj_o_gndsss float NOT NULL, \
            adj_o_headssa float NOT NULL, \
            adj_o_headsss float NOT NULL, \
            adj_o_kd float NOT NULL, \
            adj_o_legssa float NOT NULL, \
            adj_o_legsss float NOT NULL, \
            adj_o_pas float NOT NULL, \
            adj_o_rev float NOT NULL, \
            adj_o_ssa float NOT NULL, \
            adj_o_sss float NOT NULL, \
            adj_o_sub float NOT NULL, \
            adj_o_tda float NOT NULL, \
            adj_o_tds float NOT NULL, \
            adj_o_tsa float NOT NULL, \
            adj_o_tss float NOT NULL, \
            CONSTRAINT adj_stats_fighter_fkey FOREIGN KEY (fighter_id) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT adj_stats_bout_fkey FOREIGN KEY (bout_id) \
            REFERENCES ufc.bouts (bout_id), \
            CONSTRAINT adj_stats_uq UNIQUE (fighter_id, bout_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            ]
 
  
adj_avg_stats = ['DROP TABLE IF EXISTS ufc.adj_avg_stats;',
           
           'CREATE TABLE ufc.adj_avg_stats \
            (fighter_id varchar COLLATE pg_catalog."default" NOT NULL, \
             bout_id varchar COLLATE pg_catalog."default" NOT NULL, \
            adj_avg_d_bodyssa float NOT NULL, \
            adj_avg_d_bodysss float NOT NULL, \
            adj_avg_d_clinssa float NOT NULL, \
            adj_avg_d_clinsss float NOT NULL, \
            adj_avg_d_disssa float NOT NULL, \
            adj_avg_d_dissss float NOT NULL, \
            adj_avg_d_gndssa float NOT NULL, \
            adj_avg_d_gndsss float NOT NULL, \
            adj_avg_d_headssa float NOT NULL, \
            adj_avg_d_headsss float NOT NULL, \
            adj_avg_d_kd float NOT NULL, \
            adj_avg_d_legssa float NOT NULL, \
            adj_avg_d_legsss float NOT NULL, \
            adj_avg_d_pas float NOT NULL, \
            adj_avg_d_rev float NOT NULL, \
            adj_avg_d_ssa float NOT NULL, \
            adj_avg_d_sss float NOT NULL, \
            adj_avg_d_sub float NOT NULL, \
            adj_avg_d_tda float NOT NULL, \
            adj_avg_d_tds float NOT NULL, \
            adj_avg_d_tsa float NOT NULL, \
            adj_avg_d_tss float NOT NULL, \
            adj_avg_o_bodyssa float NOT NULL, \
            adj_avg_o_bodysss float NOT NULL, \
            adj_avg_o_clinssa float NOT NULL, \
            adj_avg_o_clinsss float NOT NULL, \
            adj_avg_o_disssa float NOT NULL, \
            adj_avg_o_dissss float NOT NULL, \
            adj_avg_o_gndssa float NOT NULL, \
            adj_avg_o_gndsss float NOT NULL, \
            adj_avg_o_headssa float NOT NULL, \
            adj_avg_o_headsss float NOT NULL, \
            adj_avg_o_kd float NOT NULL, \
            adj_avg_o_legssa float NOT NULL, \
            adj_avg_o_legsss float NOT NULL, \
            adj_avg_o_pas float NOT NULL, \
            adj_avg_o_rev float NOT NULL, \
            adj_avg_o_ssa float NOT NULL, \
            adj_avg_o_sss float NOT NULL, \
            adj_avg_o_sub float NOT NULL, \
            adj_avg_o_tda float NOT NULL, \
            adj_avg_o_tds float NOT NULL, \
            adj_avg_o_tsa float NOT NULL, \
            adj_avg_o_tss float NOT NULL, \
            CONSTRAINT adj_avg_stats_fighter_fkey FOREIGN KEY (fighter_id) \
            REFERENCES ufc.fighters (fighter_id), \
            CONSTRAINT adj_avg_stats_bout_fkey FOREIGN KEY (bout_id) \
            REFERENCES ufc.bouts (bout_id), \
            CONSTRAINT adj_avg_stats_uq UNIQUE (fighter_id, bout_id)) \
            WITH (OIDS = FALSE) \
            TABLESPACE pg_default;'
            ]


create_tables = {}
create_tables['fights'] = fights
create_tables['countries'] = countries
create_tables['states'] = states
create_tables['cities'] = cities
create_tables['fighters'] = fighters
create_tables['bouts'] = bouts
create_tables['methods'] = methods
create_tables['weights'] = weights
create_tables['bout_results'] = bout_results
create_tables['bout_stats'] = bout_stats
create_tables['bout_fighter_xref'] = bout_fighter_xref
create_tables['bout_corners'] = bout_corners
create_tables['bout_predictions'] = bout_predictions
create_tables['avg_stats'] = avg_stats
create_tables['adj_stats'] = adj_stats
create_tables['adj_avg_stats'] = adj_avg_stats