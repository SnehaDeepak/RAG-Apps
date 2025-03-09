// ****************************************************
// ********    MLFLOW RELATED JS       ****************
// ****************************************************

$(document).ready(function () {
    $("#deploy_runid_bank_form #sort_selection_bank_mlflow").change(function () {
        var selectedValue = $("#sort_selection_bank_mlflow option:selected").val();
        console.log(selectedValue)
        $.ajax({
            url: '/get_runs_experiments_deploy',
            data: {
                'selected_value': selectedValue
            },
            dataType: 'json',
            success: function (data) {
                console.log(data);
                //console.log(data.current_run_id_bank);
                // clear the table before adding the new rows
                $('.table_runid_data_bank tbody tr').remove();
                // add new rows to the table based on the sorted data
                $.each(data.run_details_all_dict_bank, function (i, row) {
                    console.log(row.Run_id)
                    
                    var tr = $('<tr>').append(
                        $('<td nowrap="nowrap">').html('<input type="radio" name="run_id_bank" value="' + row.Run_id + '">'),
                        $('<td nowrap="nowrap">').text(row.Run_id),
                        $('<td nowrap="nowrap">').text(row.Run_Name),
                        $('<td nowrap="nowrap">').text(row.Created),
                        $('<td nowrap="nowrap">').text(row["Duration(MM:SS)"]),
                        $('<td nowrap="nowrap">').text(row.granularity),
                        $('<td nowrap="nowrap">').text(row.chunk_size),
                        $('<td nowrap="nowrap">').text(row.sentence_window_size),
                        $('<td nowrap="nowrap">').text(row.Embeddings),
                        $('<td nowrap="nowrap">').text(row.similarity_top_k),
                        $('<td nowrap="nowrap">').text(row.rerank_top_n),
                        $('<td nowrap="nowrap">').text(row.LLM),
                        $('<td nowrap="nowrap">').text(row.temperature),
                        $('<td nowrap="nowrap">').text(row.context_window),
                        $('<td nowrap="nowrap">').text(row.max_new_tokens),
                        $('<td nowrap="nowrap">').text(row.faithfulness),
                        $('<td nowrap="nowrap">').text(row.relevancy),
                        $('<td nowrap="nowrap">').text(row["Hit Rate"]),
                        $('<td nowrap="nowrap">').text(row.MRR),
                        $('<td nowrap="nowrap">').text(row["Eval-LLM"]),
                    );
                    //if (row.Run_id == data.current_run_id_bank) {
                        // tr.addClass('highlight-deployed');
                    //    tr.css({ 'background-color': '#ffd9b0', 'color': '#094F5B' });
                    //}
                    $('.table_runid_data_bank tbody').append(tr);
                });
            },
            error: function (jqXHR, textStatus, errorThrown) {
                console.log("AJAX error:", textStatus, errorThrown);
            }
        });
    });

    document.querySelector('#deploy_runid_bank_form input[id="deploy_runid_bank"]').addEventListener('click', function () {
        var selected_run_id = document.querySelector('input[name="run_id_bank"]:checked').value;
        fetch('/set_deployed_run_id_bank', {
            method: 'POST',
            body: JSON.stringify({ 'deployed_run_id': selected_run_id }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(response => response.json())
            .then(data => {
                console.log(data)
                var current_run_id_bank = data.current_run_id_bank;
                console.log(current_run_id_bank)
                $('tr').css('background-color', '');
                $('input[value="' + current_run_id_bank + '"]').closest('tr').css({ 'background-color': '#ffd9b0', 'color': '#094F5B' });
                $('#current_run_id_bank_display').text(current_run_id_bank);
            });
    });

});
