#GETTO:

################################################################################
#                          FEATURE ENGINEERING STRATEGY                        #
################################################################################
# 1. Extract Time Parts: 
#    Turn 'Start_Time' into Hour, DayOfWeek, and Month. 
#    Rationale: Accidents at 8:00 AM on Monday (commuter rush) represent 
#    very different patterns and risks than 8:00 AM on Sunday.
#
# 2. Binary Flags: 
#    Convert the logical TRUE/FALSE columns (e.g., Junction, Crossing, 
#    Traffic_Signal) into numeric 1/0 values. 
#    Rationale: Most machine learning algorithms (like XGBoost or GLMs) 
#    require numeric inputs to process categorical predictors effectively.
#
# 3. Coordinate Rounding: 
#    If performing classification or spatial grouping, round 'Start_Lat' 
#    and 'Start_Lng' to 2 decimal places. 
#    Rationale: This creates "neighborhood grids" (approx. 1.1km sq), 
#    helping the model identify localized spatial "hot zones" rather than 
#    treating every unique GPS coordinate as a distinct point.