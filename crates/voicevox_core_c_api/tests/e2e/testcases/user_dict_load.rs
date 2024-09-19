// ユーザー辞書の登録によって読みが変化することを確認するテスト。
// 辞書ロード前後でAudioQueryのkanaが変化するかどうかで確認する。

use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::sync::LazyLock;

use assert_cmd::assert::AssertResult;
use cstr::cstr;
use libloading::Library;
use serde::{Deserialize, Serialize};
use test_util::c_api::{self, CApi, VoicevoxInitializeOptions, VoicevoxResultCode};
use test_util::OPEN_JTALK_DIC_DIR;

use crate::{
    assert_cdylib::{self, case, Utf8Output},
    snapshots,
};

case!(TestCase);

#[derive(Serialize, Deserialize)]
struct TestCase;

#[typetag::serde(name = "user_dict_load")]
impl assert_cdylib::TestCase for TestCase {
    unsafe fn exec(&self, lib: Library) -> anyhow::Result<()> {
        let lib = CApi::from_library(lib)?;

        let dict = lib.voicevox_user_dict_new();

        let mut word_uuid = [0u8; 16];

        let word = {
            let mut word = lib.voicevox_user_dict_word_make(
                cstr!("this_word_should_not_exist_in_default_dictionary").as_ptr(),
                cstr!("アイウエオ").as_ptr(),
            );
            word.word_type =
                c_api::VoicevoxUserDictWordType_VOICEVOX_USER_DICT_WORD_TYPE_PROPER_NOUN;
            word.priority = 10;

            word
        };

        assert_ok(lib.voicevox_user_dict_add_word(dict, &word, &mut word_uuid));

        let model = {
            let mut model = MaybeUninit::uninit();
            assert_ok(lib.voicevox_voice_model_file_open(
                c_api::SAMPLE_VOICE_MODEL_FILE_PATH.as_ptr(),
                model.as_mut_ptr(),
            ));
            model.assume_init()
        };

        let onnxruntime = {
            let mut onnxruntime = MaybeUninit::uninit();
            assert_ok(lib.voicevox_onnxruntime_load_once(
                lib.voicevox_make_default_load_onnxruntime_options(),
                onnxruntime.as_mut_ptr(),
            ));
            onnxruntime.assume_init()
        };

        let openjtalk = {
            let mut openjtalk = MaybeUninit::uninit();
            let open_jtalk_dic_dir = CString::new(OPEN_JTALK_DIC_DIR).unwrap();
            assert_ok(
                lib.voicevox_open_jtalk_rc_new(open_jtalk_dic_dir.as_ptr(), openjtalk.as_mut_ptr()),
            );
            openjtalk.assume_init()
        };

        let synthesizer = {
            let mut synthesizer = MaybeUninit::uninit();
            assert_ok(lib.voicevox_synthesizer_new(
                onnxruntime,
                openjtalk,
                VoicevoxInitializeOptions {
                    acceleration_mode:
                        c_api::VoicevoxAccelerationMode_VOICEVOX_ACCELERATION_MODE_CPU,
                    ..lib.voicevox_make_default_initialize_options()
                },
                synthesizer.as_mut_ptr(),
            ));
            synthesizer.assume_init()
        };

        assert_ok(lib.voicevox_synthesizer_load_voice_model(synthesizer, model));

        let mut audio_query_without_dict = std::ptr::null_mut();
        assert_ok(lib.voicevox_synthesizer_create_audio_query(
            synthesizer,
            cstr!("this_word_should_not_exist_in_default_dictionary").as_ptr(),
            STYLE_ID,
            &mut audio_query_without_dict,
        ));
        let audio_query_without_dict = serde_json::from_str::<serde_json::Value>(
            CStr::from_ptr(audio_query_without_dict).to_str()?,
        )?;

        assert_ok(lib.voicevox_open_jtalk_rc_use_user_dict(openjtalk, dict));

        let mut audio_query_with_dict = std::ptr::null_mut();
        assert_ok(lib.voicevox_synthesizer_create_audio_query(
            synthesizer,
            cstr!("this_word_should_not_exist_in_default_dictionary").as_ptr(),
            STYLE_ID,
            &mut audio_query_with_dict,
        ));

        let audio_query_with_dict = serde_json::from_str::<serde_json::Value>(
            CStr::from_ptr(audio_query_with_dict).to_str()?,
        )?;

        assert_ne!(
            audio_query_without_dict.get("kana"),
            audio_query_with_dict.get("kana")
        );

        lib.voicevox_voice_model_file_close(model);
        lib.voicevox_open_jtalk_rc_delete(openjtalk);
        lib.voicevox_synthesizer_delete(synthesizer);
        lib.voicevox_user_dict_delete(dict);

        return Ok(());

        fn assert_ok(result_code: VoicevoxResultCode) {
            std::assert_eq!(c_api::VoicevoxResultCode_VOICEVOX_RESULT_OK, result_code);
        }
        const STYLE_ID: u32 = 0;
    }

    fn assert_output(&self, output: Utf8Output) -> AssertResult {
        output
            .mask_timestamps()
            .mask_onnxruntime_version()
            .mask_windows_video_cards()
            .assert()
            .try_success()?
            .try_stdout("")?
            .try_stderr(&*SNAPSHOTS.stderr)
    }
}

static SNAPSHOTS: LazyLock<Snapshots> = snapshots::section!(user_dict_load);

#[derive(Deserialize)]
struct Snapshots {
    #[serde(deserialize_with = "snapshots::deserialize_platform_specific_snapshot")]
    stderr: String,
}
