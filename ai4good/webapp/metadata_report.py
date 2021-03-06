from datetime import datetime
import dash_html_components as html
from ai4good.utils.logger_util import get_logger

logger = get_logger(__name__)


def GenerateMetadataDict(model_id, camp, profile, model_runner, additional_metadata = None):
    current_time = datetime.now() #Get the current time (time the report is generated).
    history_df = model_runner.history_df() #Get the history of the model runner.
    try:
        if "Details" in history_df.columns and "Key" in history_df.columns and "Time" in history_df.columns: #Check if the history_df actually contains any results that can be used.
            successes_df = history_df.drop(history_df[history_df.Details != "Success"].index) #Only keep rows where a model has finished running.
            current_model_successes_df = successes_df.drop(successes_df[successes_df.Key != str((model_id, profile, camp))].index).reset_index() #Only keep rows which meet the parameters of the report.
            model_finish_time = current_model_successes_df["Time"][0] #Select the most recent model run (the report page only shows the most recent) finish time.
            try:
                version_date = current_model_successes_df["Version Date"][0] # Select the most recent model run (the report page only shows the most recent) finish time.
            # Avoids error when generating a report that was not generated to store the version date
            except:
                version_date = "Unavailable"

            started_df = history_df.drop(history_df[history_df.Status != "ModelRunResult.RUNNING"].index) #May be used in future to display the time the model was started.
        else:
            started_df = "Unknown" #No results in the history_df.
    except:
        started_df = "Unknown" #Some other error occured when trying to find the model finish time.
    metadata = {"Model ID": model_id, "Camp": camp, "Profile": profile, "Time Model Finished": model_finish_time, "Time Report Generated": current_time.strftime("%Y-%m-%d %H:%M:%S.%f"), "Version Date": version_date} #Add the metadata to a dictionary.
    if type(additional_metadata) is dict and additional_metadata != None:
        for key, value in dict.items():
            metadata[key] = value #Add any additional metadata to the dictionary which was passed into the function.
    return metadata
    
def GenerateMetadataHTML(metadata):
    html_elements = []
    if type(metadata) is not dict:
        logger.error(f"Invalid dictionary. Cannot generate metadata HTML. metadata = {metadata}. Type: {type(metadata)}.") #Handle an invalid dictionary parameter.
        return html.Div("Unable to load metadata.", style={"color": "gray", "font-size": 11, "font-style": "italic"})
    counter = 0
    for key, value in metadata.items(): #Generate the html and css.
        if counter > 0:
            html_elements.append(", ") #Add a comma between each metadata entry.
        html_elements.append(html.Span(key, style={"font-weight": "bold"})) #Embolden the metadata type.
        html_elements.append(f": {value}") #Add the metadata value.
        counter += 1
        
    return html.Div(html_elements, style={"color": "gray", "font-size": 11, "font-style": "italic"}) #Return Div to be displayed.