//object_recognition
var yolo_type_count_limit = 10;
var yolo_type_count = 0;
var yolo_type_val = 'model_0';
var yolo_current_id = 0;
var yolo_args = "";
//create btn
const YOLO_BTN_PREFIX = "yolo_";
const YOLO_BTN_SUFFIX = "_cb()";
const YOLO_MODEL_LIMIT = {'tar': "tar" ,'m5m':'m5m'};
var yolo_btn_array = ["add", "run"];
var yolo_last_model_path ;
function ObjectRecongnition(ts, id, t_name) {
    if (isClickSameFunc(id)) {
        return;
    }
    loadStreamTemplate(ts);
    postUnitV2Func(id, t_name);
    generate_button(yolo_btn_array,YOLO_BTN_PREFIX,YOLO_BTN_SUFFIX);
    ajax_interval = setInterval(function(){
    $.ajax({
        type:"post",
        url:"/data_from_device",
        dataType:"json",
        success:function(res){
            if(res != null && res.running=="Object Recongnition" ){
                if(res.models != null){
                    $("#dynamic_func_area").html('');
                    yolo_type_count = 0;
                    yolo_last_model_path = res.running_model;
                    for(let i =0 ;i<res.models.length;i++){
                        yolo_btn_add_cb(res.models[i]);
                    }
                }
                clearInterval(ajax_interval);
            }else{
                if(res.running == "error"){
                    clearInterval(ajax_interval);
                }
                yolo_type_count = 0;
                yolo_type_val = 'ob_0';
            }
        },
        error:function(res){
            yolo_type_count = 0;
            yolo_type_val = 'ob_0';
        }
    });
    },500)
}



function yolo_btn_add_cb(model_value){
    if (yolo_type_count < yolo_type_count_limit) {
        var div = document.createElement("div");
        div.className = "flex-row trainning-input yolo_type_model_div"
        div.innerHTML =
                '<button type="button" class="btn-trainning"  onclick="yolo_type_input_selected(this)" ></button>'
                +'<span class="yolo_type_model_span">'+(model_value==null?"null":model_value)+'</span>'
                +'<label class="yolo_type_model_label yolo_type_model_label_'+yolo_type_count+'" for="yolo_type_model_'+ yolo_type_count +'">upload'
                +'<svg t="1616064118468" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="8526" width="200" height="200"><path d="M512 0a512 512 0 1 0 512 512A512 512 0 0 0 512 0z m253.44 509.44a51.2 51.2 0 0 1-72.192 0L563.2 379.392V768a51.2 51.2 0 0 1-102.4 0V379.392L330.752 509.44a51.2 51.2 0 0 1-72.192-72.704l217.088-217.088a51.2 51.2 0 0 1 10.24-6.656 51.2 51.2 0 0 1 6.656-4.096 51.2 51.2 0 0 1 39.424 0 51.2 51.2 0 0 1 6.656 4.096 51.2 51.2 0 0 1 10.24 6.656l217.088 217.088a51.2 51.2 0 0 1-0.512 72.704z" fill="#ffffff" p-id="8527"></path></svg>'
                +'</label>'
                + '<form class="yolo_type_model_form" methond="post" enctype="multipart/form-data">'
                +'<input name="file" id="yolo_type_model_'+yolo_type_count+'" class="yolo_type_model_input" type="file" onchange="yolo_type_model_name_change(this)" value=' + "shape_" + yolo_type_count + '>'
                +'</form>'

        document.querySelector("#dynamic_func_area").appendChild(div);
        yolo_type_count++;

        if(yolo_last_model_path !=null && model_value!=null){
            if(yolo_last_model_path.indexOf(model_value)>0){
                $(div).find('button').click();
            }
        }
    }
}

function yolo_type_input_selected(e) {
    // $(e).parent().find('.yolo_type_model_input')[0].files[0].name;
    yolo_type_val = $(e).parent().find('.yolo_type_model_span').html();
    yolo_args = yolo_type_val
    yolo_btn_selected_mutually_exclusive(e);
}
function yolo_type_model_name_change(e) {
    let file_name = $(e)[0].files[0].name;
    if(!judge_model_suffix(file_name,YOLO_MODEL_LIMIT)){
        return;
    }
    let formData =  new FormData($(e).parent().parent().find('.yolo_type_model_form')[0])
    if(confirm("Confirm upload model?")){
        yolo_type_model_upload(file_name,$(e).parent().parent().children('span'),e,formData);
    }
}
function yolo_type_del_input(e) {
    var del_btn = document.createElement("button")
    del_btn.className = "btn-trainning-del"
    del_btn.setAttribute('onclick', 'yolo_model_remove(this)')
    e.parentNode.appendChild(del_btn);
}
function yolo_model_input_remove(e) {
    e.parentNode.remove()
    yolo_type_count--;
    yolo_type_val = '';
}

function yolo_btn_selected_mutually_exclusive(e) {
    var dom = document.getElementsByClassName("train_active");
    if (dom.length != 0) {
        for (let i = 0; i < dom.length; i++) {
            dom[i].parentNode.lastChild.remove()
            dom[i].parentNode.firstChild.className = 'btn-trainning';
        }
    }
    e.className = 'train_active btn-trainning';
    yolo_type_del_input(e)
    let parentNodes = $(e).parent().parent().find(".trainning-input");
    for (let i = 0; i < parentNodes.length; i++) {
        if ($(parentNodes[i]).children('.train_active').length > 0) {
            yolo_current_id = i;
        }
    }
}

function yolo_type_model_upload(file_name,yolo_model_name,e,formData){
    $(".uploading").fadeIn('fast');
    $.ajax({
        type:"post",
        url: "/upload/models",
        data: formData,
            processData:false,
            contentType:false,
            xhr:function(){
                var xhr = $.ajaxSettings.xhr();
                if(xhr.upload){
                    xhr.upload.addEventListener('progress',function(e){
                        var loaded =e.loaded;
                        var total = e.total;
                        var percent = Math.floor(100*loaded/total)+"%";
                        $(".uploading-process >div").css({
                            "width": percent
                        })
                    })
                }
                return xhr;
            },
            success:function(res){
                if(res == "ok"){
                    uploading_success()

                }else{
                    uploading_failure(res.error);
                    console.log(res.error);
                }
            },
            error:function(res){
                uploading_failure(res.error)
                console.log(res.error);
            },
            complete:function(){
                request_yolo_model()
            }

    })
}

function request_yolo_model(){
    ajax_interval = setInterval(function(){
        $.ajax({
            type:"post",
            url:"/data_from_device",
            dataType:"json",
            success:function(res){
                if(res != null && res.running=="Object Recongnition" ){
                    if(res.models != null){
                        $("#dynamic_func_area").html('');
                        yolo_type_count = 0;
                        yolo_last_model_path = res.running_model;
                        for(let i =0 ;i<res.models.length;i++){
                            yolo_btn_add_cb(res.models[i]);
                        }
                    }
                    clearInterval(ajax_interval);
                }else{
                    if(res.running == "null"){
                        clearInterval(ajax_interval);
                    }
                    yolo_type_count = 0;
                    yolo_type_val = 'ob_0';
                }
            },
            error:function(res){
                yolo_type_count = 0;
                yolo_type_val = 'ob_0';
                clearInterval(ajax_interval);
            }
        });
        },500)
}